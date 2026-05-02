using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using DdRL.Plugin.Logging;
using Newtonsoft.Json;

namespace DdRL.Plugin.State;

public static class ClassDiscovery
{
    private static readonly string[] Patterns =
    {
        "Battle",
        "Combat",
        "Turn",
        "Skill",
        "Item",
        "Actor",
        "Hero",
        "Monster",
        "Party",
        "Loadout"
    };
    private static readonly string[] StrictMemberDumpPatterns =
    {
        "BattleController",
        "TurnController",
        "CombatSkill",
        "CombatItem",
        "CombatActor",
        "CombatHero",
        "CombatMonster",
        "Combat.Battle",
        "Combat.CombatBhv",
        "ActorControllerBase",
        "Combat.BattleTeams",
        "Combat.Team",
        "Actor.ActorInstance",
        "ActorControllerInput",
        "PassTurnButtonBhv",
        "SkillButtonBhv",
        "SelectableSkillButton",
        "SkillSelectionBhv",
        "SkillSelectionUtils",
        "CombatInterfaceBarUiBhv",
        "ActivateSkillSignal",
        "CharacterSheetSkillButtonBhv",
        "CombatActorHitboxBhv",
        "EventBattlePass",
        "BattleTurnOrder",
        "EventBattleIncrementTurn",
        "EventBattleRemoveTurn",
    };

    public static IReadOnlyList<string> Discover(bool dumpToDisk)
    {
        var candidates = new List<string>();
        var members = new Dictionary<string, TypeMembersDump>(StringComparer.OrdinalIgnoreCase);
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type[] types;
            try
            {
                types = asm.GetTypes();
            }
            catch (ReflectionTypeLoadException ex)
            {
                types = ex.Types.Where(t => t != null).Cast<Type>().ToArray();
            }
            catch
            {
                continue;
            }

            foreach (var type in types)
            {
                var fullName = type.FullName;
                if (string.IsNullOrWhiteSpace(fullName)) continue;
                if (Patterns.Any(p => fullName.IndexOf(p, StringComparison.OrdinalIgnoreCase) >= 0))
                {
                    candidates.Add(fullName);
                    if (ShouldDumpMembers(fullName))
                    {
                        members[fullName] = BuildMembersDump(type);
                    }
                }
            }
        }

        candidates.Sort(StringComparer.OrdinalIgnoreCase);
        candidates = candidates.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        foreach (var c in candidates)
        {
            DebugLog.Info($"ClassDiscovery candidate: {c}");
        }

        if (dumpToDisk)
        {
            Dump(candidates, members);
        }
        return candidates;
    }

    private static bool ShouldDumpMembers(string fullName)
    {
        if (StrictMemberDumpPatterns.Any(p => fullName.IndexOf(p, StringComparison.OrdinalIgnoreCase) >= 0))
        {
            return true;
        }

        // Keep battle runtime classes in dump even when naming drifts across builds.
        return fullName.IndexOf("Combat.Battle", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("ActorControllerBase", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("ActorControllerInput", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("Combat.CombatBhv", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("PassTurnButtonBhv", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("SkillButtonBhv", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("SelectableSkillButton", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("SkillSelectionBhv", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("SkillSelectionUtils", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("CombatInterfaceBarUiBhv", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("ActivateSkillSignal", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("CharacterSheetSkillButtonBhv", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("CombatActorHitboxBhv", StringComparison.OrdinalIgnoreCase) >= 0
            || fullName.IndexOf("EventBattlePass", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static TypeMembersDump BuildMembersDump(Type type)
    {
        var methods = type
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Where(m => !m.IsSpecialName)
            .Select(m => new MethodDump(
                m.Name,
                m.ReturnType.FullName ?? m.ReturnType.Name,
                m.GetParameters().Select(p => $"{(p.ParameterType.FullName ?? p.ParameterType.Name)} {p.Name}").ToArray()))
            .OrderBy(m => m.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var fields = type
            .GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Select(f => new MemberDump(f.Name, f.FieldType.FullName ?? f.FieldType.Name))
            .OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var properties = type
            .GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Select(p => new MemberDump(p.Name, p.PropertyType.FullName ?? p.PropertyType.Name))
            .OrderBy(p => p.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();

        return new TypeMembersDump(type.FullName ?? type.Name, methods, fields, properties);
    }

    private static void Dump(IReadOnlyList<string> candidates, IReadOnlyDictionary<string, TypeMembersDump> members)
    {
        try
        {
            var appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var dir = Path.Combine(appData, "DDRL");
            Directory.CreateDirectory(dir);
            var classesPath = Path.Combine(dir, "classes.discovered.json");
            var membersPath = Path.Combine(dir, "members.discovered.json");
            File.WriteAllText(classesPath, JsonConvert.SerializeObject(candidates, Formatting.Indented));
            File.WriteAllText(membersPath, JsonConvert.SerializeObject(members.Values.OrderBy(v => v.TypeName, StringComparer.OrdinalIgnoreCase), Formatting.Indented));
            DebugLog.Info($"ClassDiscovery dump written: {classesPath}");
            DebugLog.Info($"ClassDiscovery members written: {membersPath}");
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Failed to dump class discovery: {ex.Message}");
        }
    }
}

public sealed record MemberDump(string Name, string TypeName);

public sealed record MethodDump(string Name, string ReturnType, IReadOnlyList<string> Parameters);

public sealed record TypeMembersDump(
    string TypeName,
    IReadOnlyList<MethodDump> Methods,
    IReadOnlyList<MemberDump> Fields,
    IReadOnlyList<MemberDump> Properties);
