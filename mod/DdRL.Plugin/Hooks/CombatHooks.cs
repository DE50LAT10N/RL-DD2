using System;
using System.Linq;
using System.Reflection;
using DdRL.Plugin.Logging;
using HarmonyLib;

namespace DdRL.Plugin.Hooks;

public static class CombatHooks
{
    private static Harmony? _harmony;
    private static Action? _onTurnBegin;
    private static Action? _onProcessPending;
    private static Action<bool>? _onBattleEnd;
    public static object? CurrentTurnContext { get; private set; }

    public static bool Install(Action onTurnBegin, Action onProcessPending, Action<bool>? onBattleEnd = null)
    {
        _onTurnBegin = onTurnBegin;
        _onProcessPending = onProcessPending;
        _onBattleEnd = onBattleEnd;

        var types = ClassPaths.BattleControllerCandidates
            .Select(name => ClassPaths.ResolveType(name))
            .Where(t => t != null)
            .Cast<Type>()
            .ToArray();
        if (types.Length == 0)
        {
            DebugLog.Warn("Hook install skipped: battle controller class not found.");
            return false;
        }

        _harmony = new Harmony("com.rl.ddrl.hooks");
        var turnMethod = FindMethod(types, ClassPaths.MethodOnHeroTurnStartCandidates);
        if (turnMethod != null)
        {
            var postfix = typeof(CombatHooks).GetMethod(nameof(OnHeroTurnPostfix), BindingFlags.NonPublic | BindingFlags.Static);
            _harmony.Patch(turnMethod, postfix: new HarmonyMethod(postfix));
            DebugLog.Info($"Patched turn method: {turnMethod.DeclaringType?.FullName}.{turnMethod.Name}");
        }
        else
        {
            DebugLog.Warn("Turn method not found in candidates.");
        }

        var battleEndMethod = FindMethod(types, ClassPaths.MethodBattleEndCandidates);
        if (battleEndMethod != null)
        {
            var postfix = typeof(CombatHooks).GetMethod(nameof(OnBattleEndPostfix), BindingFlags.NonPublic | BindingFlags.Static);
            _harmony.Patch(battleEndMethod, postfix: new HarmonyMethod(postfix));
            DebugLog.Info($"Patched battle-end method: {battleEndMethod.DeclaringType?.FullName}.{battleEndMethod.Name}");
        }
        else
        {
            DebugLog.Warn("Battle-end method not found in candidates.");
        }

        return true;
    }

    private static void OnHeroTurnPostfix(object? __instance = null)
    {
        CurrentTurnContext = __instance;
        try { _onTurnBegin?.Invoke(); } catch (Exception ex) { DebugLog.Warn($"OnHeroTurnPostfix state callback failed: {ex.Message}"); }
        try { _onProcessPending?.Invoke(); } catch (Exception ex) { DebugLog.Warn($"OnHeroTurnPostfix dispatcher callback failed: {ex.Message}"); }
    }

    private static void OnBattleEndPostfix()
    {
        try
        {
            // Battle-end runs while DD2 is tearing down combat state. Avoid
            // reflective team/actor reads here; the Python side scores the
            // terminal result from the last stable live snapshot.
            _onBattleEnd?.Invoke(false);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"OnBattleEndPostfix callback failed: {ex.Message}");
        }
    }

    private static MethodInfo? FindMethod(Type[] types, string[] methodNames)
    {
        foreach (var type in types)
        {
            foreach (var methodName in methodNames)
            {
                var method = type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
                    .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal));
                if (method != null) return method;
            }
        }

        return null;
    }
}
