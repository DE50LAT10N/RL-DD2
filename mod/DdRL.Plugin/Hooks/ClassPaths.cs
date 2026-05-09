// Central list of DD2 class and method names used by reflection hooks.
// Keeps fragile game-type discovery in one place for dispatcher/state reader logic.
// Requires updates when DD2 internals change.

using System;
using System.Linq;

namespace DdRL.Plugin.Hooks;

public static class ClassPaths
{
    public static readonly string[] BattleControllerCandidates =
    {
        "Assets.Code.Combat.CombatBhv",
        "Assets.Code.Combat.Battle",
        "Assets.Code.Actor.ActorController.ActorControllerBase",
        "Assets.Code.Combat.CombatManager",
    };

    public static readonly string[] MethodOnHeroTurnStartCandidates =
    {
        "OnStartTurn",
        "OnInTurnSelect",
        "OnTurnSelect",
        "HandleEventBattleStartRound"
    };

    public static readonly string[] MethodBattleEndCandidates =
    {
        "BattleEnd",
        "OnBattleEnd",
        "ForceEnd"
    };

    public static readonly string[] MethodPassTurnCandidates =
    {
        "OnTurnSelect",
        "OnEndTurn",
        "OnRestartTurn",
        "HandleEventBattlePass",
        "OnSkipTurn"
    };

    public static readonly string[] PassTurnButtonCandidates =
    {
        "Assets.Code.UI.PassTurnButtonBhv"
    };

    public static readonly string[] MethodPassButtonCandidates =
    {
        "OnClick",
        "OnClicked",
        "OnButtonClick",
        "OnPointerClick",
        "Click",
        "Press",
        "Pass",
        "OnPass",
        "PassTurn",
        "OnPassTurn",
        "OnEndTurn"
    };

    public static Type? ResolveType(params string[] candidates)
    {
        foreach (var name in candidates.Where(v => !string.IsNullOrWhiteSpace(v)))
        {
            var direct = Type.GetType(name, throwOnError: false);
            if (direct != null) return direct;

            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var fromAsm = asm.GetType(name, throwOnError: false, ignoreCase: false);
                if (fromAsm != null) return fromAsm;
            }
        }

        return null;
    }
}
