// Optional DD2 ScriptableObject data extractor.
// Dumps game data for improving simulator fixtures and overrides.
// Intended for research/debug runs, not normal live inference.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using DdRL.Plugin.Logging;
using Newtonsoft.Json;

namespace DdRL.Plugin.State;

public static class DataExtractor
{
    private static readonly (string Pattern, string Bucket)[] Rules =
    {
        ("HeroType", "heroes"),
        ("Skill", "heroes"),
        ("MonsterType", "monsters"),
        ("Token", "tokens"),
        ("CombatItem", "items"),
        ("Encounter", "encounters"),
    };

    public static int DumpAllScriptableObjects()
    {
        var soType = Type.GetType("UnityEngine.ScriptableObject, UnityEngine.CoreModule", throwOnError: false);
        var resourcesType = Type.GetType("UnityEngine.Resources, UnityEngine.CoreModule", throwOnError: false);
        if (soType == null || resourcesType == null)
        {
            DebugLog.Warn("DataExtractor skipped: UnityEngine types unavailable.");
            return 0;
        }

        var dumpRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "DDRL", "data");
        foreach (var (_, bucket) in Rules) Directory.CreateDirectory(Path.Combine(dumpRoot, bucket));

        var allObjects = new List<object>();
        var findMethod = resourcesType.GetMethods(BindingFlags.Public | BindingFlags.Static)
            .FirstOrDefault(m => m.Name == "FindObjectsOfTypeAll" && m.IsGenericMethodDefinition && m.GetParameters().Length == 0);
        if (findMethod != null)
        {
            var generic = findMethod.MakeGenericMethod(soType);
            var result = generic.Invoke(null, null);
            if (result is IEnumerable enumerable)
            {
                foreach (var obj in enumerable)
                {
                    if (obj != null) allObjects.Add(obj);
                }
            }
        }

        // Addressables fallback (best effort)
        var addrType = Type.GetType("UnityEngine.AddressableAssets.Addressables, Unity.Addressables", throwOnError: false);
        if (addrType != null) DebugLog.Info("Addressables detected; using ScriptableObject dump + post overrides if needed.");

        var count = 0;
        foreach (var obj in allObjects)
        {
            var typeName = obj.GetType().Name;
            var bucket = ResolveBucket(typeName);
            if (bucket == null) continue;

            var payload = ReflectiveSerialize(obj, depth: 0, seen: new HashSet<object>(ReferenceEqualityComparer.Instance));
            var id = SanitizeId(ReadId(obj, typeName));
            var withMetadata = EnsureMetadata(payload, id);
            var path = Path.Combine(dumpRoot, bucket, $"{id}.json");
            var json = JsonConvert.SerializeObject(withMetadata, Formatting.Indented);
            File.WriteAllText(path, json);
            count++;
        }

        DebugLog.Info($"DataExtractor dumped {count} records to {dumpRoot}");
        return count;
    }

    private static Dictionary<string, object?> EnsureMetadata(object? payload, string id)
    {
        if (payload is Dictionary<string, object?> dict)
        {
            if (!dict.ContainsKey("id")) dict["id"] = id;
            if (!dict.ContainsKey("name")) dict["name"] = id;
            if (!dict.ContainsKey("skills")) dict["skills"] = new List<object>();
            return dict;
        }
        return new Dictionary<string, object?> { ["id"] = id, ["name"] = id, ["skills"] = new List<object>(), ["raw"] = payload };
    }

    private static string? ResolveBucket(string typeName)
    {
        foreach (var (pattern, bucket) in Rules)
            if (typeName.IndexOf(pattern, StringComparison.OrdinalIgnoreCase) >= 0) return bucket;
        return null;
    }

    private static string ReadId(object obj, string fallback)
    {
        var t = obj.GetType();
        foreach (var name in new[] { "Id", "ID", "name", "Name", "m_Name" })
        {
            var p = t.GetProperty(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (p != null)
            {
                var v = p.GetValue(obj)?.ToString();
                if (!string.IsNullOrWhiteSpace(v)) return v!;
            }
            var f = t.GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (f != null)
            {
                var v = f.GetValue(obj)?.ToString();
                if (!string.IsNullOrWhiteSpace(v)) return v!;
            }
        }
        return fallback;
    }

    private static object? ReflectiveSerialize(object? value, int depth, HashSet<object> seen)
    {
        if (value == null) return null;
        if (depth > 4) return value.ToString();
        var type = value.GetType();
        if (type.IsPrimitive || value is string || value is decimal) return value;
        if (value is Enum) return value.ToString();
        if (!type.IsValueType)
        {
            if (seen.Contains(value)) return "<cycle>";
            seen.Add(value);
        }
        if (value is IEnumerable e && value is not string)
        {
            var list = new List<object?>();
            foreach (var item in e) list.Add(ReflectiveSerialize(item, depth + 1, seen));
            return list;
        }
        var dict = new Dictionary<string, object?>();
        foreach (var p in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (p.GetIndexParameters().Length > 0) continue;
            try { dict[p.Name] = ReflectiveSerialize(p.GetValue(value), depth + 1, seen); } catch { }
        }
        foreach (var f in type.GetFields(BindingFlags.Public | BindingFlags.Instance))
        {
            try { dict[f.Name] = ReflectiveSerialize(f.GetValue(value), depth + 1, seen); } catch { }
        }
        return dict.Count > 0 ? dict : value.ToString();
    }

    private static string SanitizeId(string id)
    {
        var cleaned = new string(id.Select(ch => char.IsLetterOrDigit(ch) || ch == '_' || ch == '-' ? ch : '_').ToArray());
        return string.IsNullOrWhiteSpace(cleaned) ? "unknown" : cleaned;
    }
}

internal sealed class ReferenceEqualityComparer : IEqualityComparer<object>
{
    public static readonly ReferenceEqualityComparer Instance = new();
    public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
    public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
}
