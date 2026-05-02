using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using DdRL.Plugin.Hooks;
using DdRL.Plugin.Ipc;
using DdRL.Plugin.Logging;

namespace DdRL.Plugin.Actions;

public sealed class Dispatcher
{
    private readonly JsonLineServer _server;
    private readonly bool _commitProbeMode;
    private readonly bool _enableSkillHooks;
    private readonly ConcurrentQueue<PendingAction> _pending = new();
    public int PendingCount => _pending.Count;
    private static bool _loggedActorControllerMethods;

    public Dispatcher(JsonLineServer server, bool commitProbeMode, bool enableSkillHooks)
    {
        _server = server;
        _commitProbeMode = commitProbeMode;
        _enableSkillHooks = enableSkillHooks;
    }

    public void EnqueueAction(int requestId, int heroSlot, int? skillIdx, int? targetIdx, string? itemId, bool passTurn, string? targetTeam = null, int? moveDelta = null)
    {
        _pending.Enqueue(new PendingAction(requestId, heroSlot, skillIdx, targetIdx, itemId, passTurn, targetTeam, moveDelta));
    }

    public void ProcessPending()
    {
        var processed = 0;
        while (_pending.TryDequeue(out var action))
        {
            processed++;
            DebugLog.Info($"Processing action request_id={action.RequestId} pass_turn={action.PassTurn} commit_probe={_commitProbeMode}");
            if (TryExecute(action, _commitProbeMode, out var reason, out var method))
            {
                DebugLog.Info($"Action success request_id={action.RequestId} method={method}");
                _server.Send(Protocol.MakeAck(action.RequestId, ok: true, method: method));
            }
            else
            {
                DebugLog.Warn($"Action failed request_id={action.RequestId} reason={reason ?? "execute_failed"}");
                _server.Send(Protocol.MakeAck(action.RequestId, ok: false, reason: reason ?? "execute_failed"));
            }
        }
        if (processed > 0)
        {
            DebugLog.Info($"Processed action batch count={processed}");
        }
    }

    private bool TryExecute(PendingAction action, bool commitProbeMode, out string? reason, out string methodName)
    {
        reason = null;
        methodName = "hook";

        var controllerType = ClassPaths.BattleControllerCandidates
            .Select(name => ClassPaths.ResolveType(name))
            .FirstOrDefault(t => t != null);
        if (controllerType == null)
        {
            reason = "battle_controller_missing";
            return false;
        }

        var controller = CombatHooks.CurrentTurnContext ?? ResolveControllerInstance(controllerType);
        if (controller == null)
        {
            reason = "controller_instance_missing";
            return false;
        }

        try
        {
            var probeRoot = ResolveBestBattleRoot(controller) ?? controller;
            var preSig = TryBuildStateSignature(probeRoot);
            DebugLog.Info($"Action pre-state signature={preSig}");
            if (action.PassTurn)
            {
                methodName = "pass";
                var actor = ResolveCurrentActor(controller);
                var passActorController = actor != null ? ResolveActorController(actor) : null;
                if (actor != null && passActorController != null && TryExecutePassStressSkill(probeRoot, actor, passActorController, action, preSig))
                {
                    methodName = "pass_stress_skill";
                    return true;
                }
                if (actor != null && TryExecuteBattlePassEvent(probeRoot, actor, action, preSig))
                {
                    methodName = "battle_pass_event";
                    return true;
                }
                if (TryExecutePassButtonSequence(action, commitProbeMode, preSig, probeRoot, actor != null ? ResolveActorGuid(actor) : 0))
                {
                    methodName = "pass_button";
                    return true;
                }
                if (passActorController != null && TryExecutePassTurnSequence(passActorController, action, commitProbeMode, preSig, probeRoot))
                {
                    return true;
                }
                reason = "pass_turn_unavailable";
                return false;
            }

            if (action.MoveDelta.HasValue)
            {
                if (TryExecuteMove(action, probeRoot, controller, preSig, out var moveMethod, out var moveReason))
                {
                    methodName = moveMethod;
                    return true;
                }
                methodName = "move";
                reason = moveReason ?? "move_unavailable";
                return false;
            }

            var actorInstance = ResolveCurrentActor(controller);
            if (actorInstance == null)
            {
                reason = "current_actor_missing";
                return false;
            }

            var isItemAction = !string.IsNullOrWhiteSpace(action.ItemId);
            methodName = isItemAction ? "item" : "hook";

            var actorController = ResolveActorController(actorInstance);
            if (actorController != null && !_loggedActorControllerMethods)
            {
                _loggedActorControllerMethods = true;
                LogControllerMethodHints(actorController.GetType());
            }

            var skillPlan = isItemAction
                ? BuildItemPlan(probeRoot, actorInstance, actorController, action)
                : BuildSkillPlan(probeRoot, actorInstance, actorController, action);
            LogSkillDiagnostics(skillPlan, actorInstance, actorController, action);

            if (!_enableSkillHooks)
            {
                reason = "unsafe_skill_hook_disabled";
                DebugLog.Warn("Skill hook execution is disabled because direct ActorController reflection can crash DD2. Pass/turn-order actions remain enabled.");
                return false;
            }

            if (TryExecuteSelectedSkillPath(probeRoot, actorInstance, actorController, action, skillPlan, preSig))
            {
                methodName = isItemAction ? "selected_item_path" : "selected_skill_path";
                return true;
            }

            reason = isItemAction ? "selected_item_path_no_delta" : "selected_skill_path_no_delta";
            DebugLog.Warn($"{(isItemAction ? "Selected-item" : "Selected-skill")} path did not change battle state; unsafe ActorController fallback skipped.");
            return false;
        }
        catch (Exception ex)
        {
            reason = $"execute_failed: {ex.GetType().Name}: {ex.Message}";
            DebugLog.Warn($"Action invoke failed: {reason}");
            return false;
        }
    }

    private static object?[] BuildArgs(MethodInfo method, PendingAction action)
    {
        var ps = method.GetParameters();
        if (ps.Length == 0) return Array.Empty<object?>();

        var values = new List<object?>(ps.Length);
        foreach (var p in ps)
        {
            values.Add(ConvertArgumentValue(p.ParameterType, action));
        }

        return values.ToArray();
    }

    private static object? ResolveControllerInstance(Type controllerType)
    {
        var singleton = controllerType.GetProperty("Instance", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.FlattenHierarchy);
        if (singleton != null) return singleton.GetValue(null, null);
        var field = controllerType.GetField("Instance", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.FlattenHierarchy);
        if (field != null) return field.GetValue(null);
        if (typeof(UnityEngine.Object).IsAssignableFrom(controllerType))
            return UnityEngine.Object.FindObjectOfType(controllerType);
        return null;
    }

    private static MethodInfo? FindMethod(Type? type, string[] names)
    {
        if (type == null) return null;
        var methods = type
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
            .Where(m => names.Any(name => string.Equals(m.Name, name, StringComparison.Ordinal)))
            .Where(IsInvokableMethod)
            .ToList();

        return methods.FirstOrDefault();
    }

    private static object? ResolveCurrentActor(object controller)
    {
        var battleRoot = controller.GetType().GetField("m_Battle", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(controller)
            ?? controller.GetType().GetProperty("Battle", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(controller);
        if (battleRoot != null)
        {
            controller = battleRoot;
        }

        var directActor = controller.GetType().GetField("m_PerformerActor", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(controller)
            ?? controller.GetType().GetProperty("PerformerActor", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(controller);
        if (directActor != null) return directActor;

        var guidMember = controller.GetType().GetProperty("CurrentActorGuid", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var guidRaw = guidMember?.GetValue(controller);
        if (guidRaw == null) return null;

        var teams = controller.GetType().GetField("m_BattleTeams", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(controller)
            ?? controller.GetType().GetProperty("BattleTeams", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(controller);
        var getActor = teams?.GetType().GetMethod("GetActorFromActorGuid", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        return getActor?.Invoke(teams, new[] { guidRaw });
    }

    private static object? ResolveActorController(object actorInstance)
    {
        return actorInstance.GetType().GetProperty("Controller", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(actorInstance)
            ?? actorInstance.GetType().GetField("m_ActorController", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(actorInstance);
    }

    private static object? ConvertArgumentValue(Type targetType, PendingAction action)
    {
        if (targetType == typeof(string))
        {
            return action.ItemId ?? string.Empty;
        }

        var numeric = action.TargetIdx ?? action.SkillIdx ?? action.HeroSlot;
        if (targetType.IsEnum)
        {
            return Enum.ToObject(targetType, numeric);
        }

        if (targetType == typeof(int)) return numeric;
        if (targetType == typeof(uint)) return (uint)Math.Max(0, numeric);
        if (targetType == typeof(bool)) return action.PassTurn;
        if (targetType == typeof(float)) return (float)numeric;
        if (targetType == typeof(double)) return (double)numeric;
        if (targetType == typeof(byte)) return (byte)Math.Max(0, Math.Min(255, numeric));
        if (targetType == typeof(short)) return (short)numeric;

        // Event objects and other complex payloads are not constructible safely.
        throw new InvalidOperationException($"unsupported_parameter_type:{targetType.FullName ?? targetType.Name}");
    }

    private static bool IsInvokableMethod(MethodInfo method)
    {
        if (method.IsAbstract) return false;
        var ps = method.GetParameters();
        foreach (var p in ps)
        {
            var t = p.ParameterType;
            var name = t.FullName ?? t.Name;
            if (name.Contains(".Events.Event", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }
        return true;
    }

    private static string DescribeMethod(MethodInfo method)
    {
        var ps = method.GetParameters()
            .Select(p => $"{(p.ParameterType.FullName ?? p.ParameterType.Name)} {p.Name}");
        return $"{method.DeclaringType?.FullName}.{method.Name}({string.Join(", ", ps)})";
    }

    private static bool TryExecuteControllerInputSequence(object actorController, PendingAction action, bool commitProbeMode, string preSig, object probeRoot)
    {
        if (action.PassTurn) return false;
        var t = actorController.GetType();
        if ((t.FullName ?? string.Empty).IndexOf("ActorControllerInput", StringComparison.OrdinalIgnoreCase) < 0)
        {
            return false;
        }

        var selectSkill = t.GetMethod("AttemptSelectSkill", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var selectTarget = t.GetMethod("SelectTargetGuid", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var attemptTarget = t.GetMethod("AttemptSelectTarget", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (selectSkill == null || attemptTarget == null)
        {
            return false;
        }

        try
        {
            var lastSig = preSig;
            // 1) Arm current skill selection in controller.
            InvokeWithProbe(actorController, selectSkill, action, "probe.AttemptSelectSkill", commitProbeMode, lastSig, out lastSig, probeRoot);

            // 2) Best-effort move target selector to requested index by cycling.
            //    If API is no-arg only, this still converges to some valid target.
            if (selectTarget != null && action.TargetIdx.HasValue)
            {
                var hops = Math.Max(0, action.TargetIdx.Value);
                for (var i = 0; i < hops; i++)
                {
                    InvokeWithProbe(actorController, selectTarget, action, $"probe.SelectTargetGuid[{i}]", commitProbeMode, lastSig, out lastSig, probeRoot);
                }
            }

            // 3) Confirm target selection.
            InvokeWithProbe(actorController, attemptTarget, action, "probe.AttemptSelectTarget", commitProbeMode, lastSig, out lastSig, probeRoot);

            // 4) Finalize turn selection when controller requires explicit confirm.
            var onTurnSelect = t.GetMethod("OnTurnSelect", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (onTurnSelect != null)
            {
                InvokeWithProbe(actorController, onTurnSelect, action, "probe.OnTurnSelect", commitProbeMode, lastSig, out lastSig, probeRoot);
            }
            var onEndTurn = t.GetMethod("OnEndTurn", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (onEndTurn != null)
            {
                InvokeWithProbe(actorController, onEndTurn, action, "probe.OnEndTurn", commitProbeMode, lastSig, out lastSig, probeRoot);
            }
            DebugLog.Info("Hook controller-input sequence executed: AttemptSelectSkill -> SelectTargetGuid* -> AttemptSelectTarget -> OnTurnSelect -> OnEndTurn");
            return HasStateDelta(preSig, lastSig);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Hook controller-input sequence failed: {ex.Message}");
            return false;
        }
    }

    private static bool TryExecuteMove(
        PendingAction action,
        object probeRoot,
        object controller,
        string preSig,
        out string methodName,
        out string? reason)
    {
        methodName = "move";
        reason = null;
        var actor = ResolveCurrentActor(controller);
        if (actor == null)
        {
            reason = "move_actor_missing";
            return false;
        }
        var actorController = ResolveActorController(actor);
        if (actorController == null)
        {
            reason = "move_controller_missing";
            return false;
        }
        var moveDelta = action.MoveDelta ?? 0;
        if (moveDelta == 0)
        {
            reason = "move_delta_zero";
            return false;
        }

        var actorRank = ReadIntValue(actor, new[] { "TeamPosition", "Slot", "Index", "Position" }, -1);
        var targetRank = actorRank + moveDelta;
        if (actorRank < 0 || targetRank < 0)
        {
            reason = "move_target_rank_invalid";
            DebugLog.Warn($"TryExecuteMove: target rank invalid actor_rank={actorRank} delta={moveDelta}.");
            return false;
        }
        var targetGuid = ResolveTargetActorGuid(probeRoot, targetRank, "heroes");
        if (targetGuid == 0)
        {
            reason = "move_target_actor_missing";
            DebugLog.Warn($"TryExecuteMove: no actor at target rank {targetRank}.");
            return false;
        }

        // Prefer move_skill_id from the action payload (StateReader provides it when known).
        // Fall back to actor reflection. If still missing, scan equipped skills for a likely
        // move marker. Worst case, we just won't have a skill_id and the commit will fail.
        var moveSkillId = !string.IsNullOrWhiteSpace(action.ItemId) ? action.ItemId : ResolveMoveSkillId(actor);
        // ItemId is reused as a transport for move_skill_id to avoid changing PendingAction shape.

        if (string.IsNullOrWhiteSpace(moveSkillId))
        {
            reason = "move_skill_id_missing";
            DebugLog.Warn("TryExecuteMove: move skill id could not be resolved.");
            return false;
        }

        RecalculateSkillTargets(actorController, moveSkillId);
        var isValidSkill = InvokeBoolMethod(actorController, "GetIsValidSkill", moveSkillId);
        var isValidTarget = InvokeBoolMethod(actorController, "GetIsValidSkillTarget", moveSkillId, targetGuid);
        // Note: validity probes are unreliable in BepInEx context; we proceed regardless.

        var equipped = ReadStringListFromMethod(actor, "GetEquippedCombatSkillIds");
        if (!equipped.Contains(moveSkillId))
        {
            equipped.Add(moveSkillId);
        }
        var moveAction = action with
        {
            SkillIdx = null,
            ItemId = null,
            PassTurn = false,
            TargetIdx = targetRank,
            TargetTeam = "heroes",
        };
        var plan = new SkillPlan(equipped, moveSkillId, targetGuid, isValidSkill, isValidTarget);
        DebugLog.Info(
            $"TryExecuteMove: invoking selected-skill path move_skill_id={moveSkillId} delta={moveDelta} " +
            $"actor_rank={actorRank} target_rank={targetRank} target_guid={targetGuid} " +
            $"valid_skill={FormatNullableBool(isValidSkill)} valid_target={FormatNullableBool(isValidTarget)}.");

        if (TryExecuteSelectedSkillPath(probeRoot, actor, actorController, moveAction, plan, preSig))
        {
            methodName = "selected_move_path";
            return true;
        }

        reason = "selected_move_path_no_delta";
        return false;
    }

    private static string? ResolveMoveSkillId(object actorInstance)
    {
        try
        {
            var baseSkillIds = ReadStringListFromMethod(actorInstance, "GetEquippedCombatSkillIds");
            var withMove = ReadEquippedSkillsIncludingMove(actorInstance, includeMoveSkill: true, includePassSkill: false);
            if (withMove != null && withMove.Count > 0)
            {
                var diff = withMove.Where(id => !baseSkillIds.Contains(id) && !string.IsNullOrWhiteSpace(id)).ToList();
                if (diff.Count > 0)
                {
                    return diff[0];
                }
            }

            // Fallback: ask the actor directly via any *Move*Skill*Id-like accessor.
            foreach (var name in new[] { "GetMoveSkillId", "GetCombatMoveSkillId", "MoveSkillId", "m_MoveSkillId" })
            {
                var member = ReadMemberValue(actorInstance, new[] { name });
                if (member != null)
                {
                    var asString = member.ToString();
                    if (!string.IsNullOrWhiteSpace(asString)) return asString;
                }
                var byMethod = InvokeStringMethod(actorInstance, name);
                if (!string.IsNullOrWhiteSpace(byMethod)) return byMethod;
            }
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"ResolveMoveSkillId failed: {ex.GetType().Name}: {ex.Message}");
        }
        return null;
    }

    private static List<object> EnumValuesFromFields(Type type)
    {
        var values = new List<object>();
        try
        {
            var fields = type.GetFields(BindingFlags.Public | BindingFlags.Static);
            foreach (var f in fields)
            {
                if (!f.IsLiteral) continue;
                try
                {
                    var v = f.GetValue(null);
                    if (v != null) values.Add(v);
                }
                catch { }
            }
        }
        catch { }
        return values;
    }

    private static List<string>? ReadEquippedSkillsIncludingMove(object actorInstance, bool includeMoveSkill, bool includePassSkill)
    {
        try
        {
            // Lenient: match by name + 3 params with last two bool. Param[0] type is invoked as enum if possible.
            var method = actorInstance.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.FlattenHierarchy)
                .FirstOrDefault(m =>
                {
                    if (!string.Equals(m.Name, "GetEquippedCombatSkillIds", StringComparison.Ordinal)) return false;
                    var ps = m.GetParameters();
                    return ps.Length == 3 && ps[1].ParameterType == typeof(bool) && ps[2].ParameterType == typeof(bool);
                });
            if (method == null)
            {
                return null;
            }
            var firstParamType = method.GetParameters()[0].ParameterType;
            var firstArgCandidates = new List<object?>();
            firstArgCandidates.AddRange(EnumValuesFromFields(firstParamType));
            if (firstArgCandidates.Count == 0 && firstParamType.IsValueType)
            {
                try { firstArgCandidates.Add(Activator.CreateInstance(firstParamType)); }
                catch { }
            }
            if (firstArgCandidates.Count == 0)
            {
                firstArgCandidates.Add(null);
            }

            foreach (var firstArg in firstArgCandidates)
            {
                object? raw;
                try
                {
                    raw = method.Invoke(actorInstance, new[] { firstArg, includeMoveSkill, includePassSkill });
                }
                catch (Exception ex)
                {
                    DebugLog.Info($"Dispatcher.ReadEquippedSkillsIncludingMove invoke failed for arg0={firstArg}: {ex.GetType().Name}: {ex.Message}");
                    continue;
                }
                if (raw is not System.Collections.IEnumerable enumerable) continue;
                var list = new List<string>();
                foreach (var item in enumerable)
                {
                    if (item == null) continue;
                    var asString = item.ToString();
                    if (!string.IsNullOrWhiteSpace(asString)) list.Add(asString);
                }
                return list;
            }
            return null;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"ReadEquippedSkillsIncludingMove failed: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static string? InvokeStringMethod(object target, string methodName)
    {
        try
        {
            var method = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == 0);
            if (method == null) return null;
            return method.Invoke(target, Array.Empty<object?>())?.ToString();
        }
        catch
        {
            return null;
        }
    }

    private static bool TryExecutePassTurnSequence(object actorController, PendingAction action, bool commitProbeMode, string preSig, object probeRoot)
    {
        var t = actorController.GetType();
        if ((t.FullName ?? string.Empty).IndexOf("ActorControllerInput", StringComparison.OrdinalIgnoreCase) < 0)
        {
            return false;
        }

        var onTurnSelect = t.GetMethod("OnTurnSelect", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var onEndTurn = t.GetMethod("OnEndTurn", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (onEndTurn == null)
        {
            return false;
        }

        try
        {
            var lastSig = preSig;
            if (onTurnSelect != null)
            {
                InvokeWithProbe(actorController, onTurnSelect, action, "probe.OnTurnSelect", commitProbeMode, lastSig, out lastSig, probeRoot);
            }
            InvokeWithProbe(actorController, onEndTurn, action, "probe.OnEndTurn", commitProbeMode, lastSig, out lastSig, probeRoot);
            DebugLog.Info("Pass controller-input sequence executed: OnTurnSelect -> OnEndTurn");
            return HasStateDelta(preSig, lastSig);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Pass controller-input sequence failed: {ex.Message}");
            return false;
        }
    }

    private static bool TryExecutePassStressSkill(object probeRoot, object actorInstance, object actorController, PendingAction action, string preSig)
    {
        const string passSkillId = "pass_stress";
        var candidates = new List<uint>();
        var actorGuid = ResolveActorGuid(actorInstance);
        if (actorGuid != 0)
        {
            candidates.Add(actorGuid);
        }
        var validEntryGuid = ResolveFirstValidTargetGuid(actorController, passSkillId);
        if (validEntryGuid != 0 && !candidates.Contains(validEntryGuid))
        {
            candidates.Add(validEntryGuid);
        }
        if (candidates.Count == 0)
        {
            DebugLog.Warn("Pass-stress skill skipped: no target guid.");
            return false;
        }

        bool? isValidSkill = null;
        bool? isValidTarget = null;
        var targetGuid = 0u;
        foreach (var candidateGuid in candidates)
        {
            isValidSkill = InvokeBoolMethod(actorController, "GetIsValidSkill", passSkillId);
            isValidTarget = InvokeBoolMethod(actorController, "GetIsValidSkillTarget", passSkillId, candidateGuid);
            if (isValidSkill != false && isValidTarget != false)
            {
                targetGuid = candidateGuid;
                break;
            }
        }
        if (targetGuid == 0)
        {
            var renderedCandidates = string.Join(",", candidates);
            DebugLog.Warn(
                $"Pass-stress skill skipped: valid_skill={FormatNullableBool(isValidSkill)} " +
                $"valid_target={FormatNullableBool(isValidTarget)} target_candidates=[{renderedCandidates}].");
            return false;
        }

        var passAction = action with { SkillIdx = null, TargetIdx = 0, TargetTeam = "heroes" };
        var equipped = ReadStringListFromMethod(actorInstance, "GetEquippedCombatSkillIds");
        if (!equipped.Contains(passSkillId))
        {
            equipped.Add(passSkillId);
        }
        var plan = new SkillPlan(equipped, passSkillId, targetGuid, isValidSkill, isValidTarget);
        DebugLog.Info($"Pass-stress skill attempt target_guid={targetGuid}");
        return TryExecuteSelectedSkillPath(probeRoot, actorInstance, actorController, passAction, plan, preSig);
    }

    private static bool TryExecuteBattlePassEvent(object probeRoot, object actorInstance, PendingAction action, string preSig)
    {
        var actorGuid = ResolveActorGuid(actorInstance);
        if (actorGuid == 0)
        {
            DebugLog.Warn("Battle-pass event skipped: actor guid missing.");
            return false;
        }

        var eventType = ClassPaths.ResolveType("Assets.Code.Combat.Events.EventBattlePass");
        if (eventType == null)
        {
            DebugLog.Warn("Battle-pass event skipped: EventBattlePass type missing.");
            return false;
        }

        try
        {
            var evt = CreateEventInstance(eventType, actorGuid);
            if (evt == null)
            {
                DebugLog.Warn("Battle-pass event skipped: EventBattlePass instance could not be created.");
                return false;
            }
            if (!TryWriteMemberValue(evt, new[] { "m_ActorGuid", "ActorGuid", "Guid" }, actorGuid))
            {
                DebugLog.Warn("Battle-pass event skipped: actor guid field missing.");
                return false;
            }
            var startInvoke = eventType.GetMethod("StartInvoke", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (startInvoke == null || startInvoke.GetParameters().Length != 0)
            {
                DebugLog.Warn("Battle-pass event skipped: StartInvoke missing.");
                return false;
            }

            startInvoke.Invoke(evt, Array.Empty<object?>());
            var postSig = TryBuildStateSignature(probeRoot);
            DebugLog.Info($"Battle-pass event StartInvoke ok actor_guid={actorGuid} delta={HasStateDelta(preSig, postSig)} pre={preSig} post={postSig}");
            if (HasStateDelta(preSig, postSig))
            {
                return true;
            }

            return TryInvokeBattlePassHandlers(probeRoot, actorInstance, evt, actorGuid, action, postSig, preSig);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Battle-pass event failed: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static bool TryInvokeBattlePassHandlers(object probeRoot, object actorInstance, object evt, uint actorGuid, PendingAction action, string currentSig, string preSig)
    {
        var targets = new List<object>();
        AddIfNotNull(targets, ReadMemberValue(probeRoot, new[] { "m_BattleStateMachine", "BattleStateMachine" }));
        AddIfNotNull(targets, actorInstance);
        AddIfNotNull(targets, ResolveActorController(actorInstance));
        AddIfNotNull(targets, FindCombatActorBhv(actorGuid));

        var invokedAny = false;
        var lastSig = currentSig;
        foreach (var target in targets)
        {
            var method = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m =>
                    string.Equals(m.Name, "HandleEventBattlePass", StringComparison.Ordinal) &&
                    m.GetParameters().Length == 1 &&
                    m.GetParameters()[0].ParameterType.IsAssignableFrom(evt.GetType()));
            if (method == null)
            {
                continue;
            }

            try
            {
                method.Invoke(target, new[] { evt });
                invokedAny = true;
                lastSig = TryBuildStateSignature(probeRoot);
                DebugLog.Info($"Battle-pass handler invoke: target={target.GetType().FullName} method={DescribeMethod(method)} delta={HasStateDelta(preSig, lastSig)} pre={preSig} post={lastSig}");
                if (HasStateDelta(preSig, lastSig))
                {
                    return true;
                }
            }
            catch (Exception ex)
            {
                DebugLog.Warn($"Battle-pass handler failed: target={target.GetType().FullName} method={DescribeMethod(method)} error={ex.GetType().Name}: {ex.Message}");
            }
        }

        if (!invokedAny)
        {
            DebugLog.Warn("Battle-pass event had no direct handlers to invoke.");
        }
        return false;
    }

    private static void AddIfNotNull(List<object> targets, object? target)
    {
        if (target == null || targets.Any(existing => ReferenceEquals(existing, target)))
        {
            return;
        }
        targets.Add(target);
    }

    private static object? CreateEventInstance(Type eventType, uint actorGuid)
    {
        try
        {
            var evt = Activator.CreateInstance(eventType, nonPublic: true);
            if (evt != null) return evt;
        }
        catch (Exception ex)
        {
            DebugLog.Info($"Event default constructor unavailable for {eventType.FullName}: {ex.GetType().Name}: {ex.Message}");
        }

        foreach (var ctor in eventType.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            try
            {
                var args = ctor.GetParameters()
                    .Select(p => BuildEventConstructorArg(p, actorGuid))
                    .ToArray();
                var evt = ctor.Invoke(args);
                if (evt != null)
                {
                    DebugLog.Info($"Event constructor selected: {DescribeConstructor(ctor)}");
                    return evt;
                }
            }
            catch (Exception ex)
            {
                DebugLog.Info($"Event constructor failed: {DescribeConstructor(ctor)} -> {ex.GetType().Name}: {ex.Message}");
            }
        }

        try
        {
            var evt = FormatterServices.GetUninitializedObject(eventType);
            DebugLog.Info($"Event instance created without constructor: {eventType.FullName}");
            return evt;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Event uninitialized allocation failed for {eventType.FullName}: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static object? BuildEventConstructorArg(ParameterInfo parameter, uint actorGuid)
    {
        var t = parameter.ParameterType;
        var name = parameter.Name ?? string.Empty;
        if (t == typeof(uint)) return actorGuid;
        if (t == typeof(int)) return name.IndexOf("amount", StringComparison.OrdinalIgnoreCase) >= 0 ? 1 : (int)actorGuid;
        if (t == typeof(bool)) return false;
        if (t == typeof(string)) return string.Empty;
        if (t.IsEnum) return Enum.ToObject(t, 0);
        if (!t.IsValueType) return null;
        return Activator.CreateInstance(t);
    }

    private static string DescribeConstructor(ConstructorInfo ctor)
    {
        var ps = ctor.GetParameters()
            .Select(p => $"{(p.ParameterType.FullName ?? p.ParameterType.Name)} {p.Name}");
        return $"{ctor.DeclaringType?.FullName}({string.Join(", ", ps)})";
    }

    private static uint ResolveActorGuid(object actorInstance)
    {
        return ReadUIntValue(actorInstance, new[] { "ActorGuid", "Guid", "m_ActorGuid", "ActorDataGuid", "DataGuid" }, 0);
    }

    private static uint ResolveFirstValidTargetGuid(object actorController, string skillId)
    {
        var entries = InvokeEnumerableMethod(actorController, "GetValidSkillTargetEntries");
        if (entries == null) return 0;

        foreach (var entry in entries)
        {
            if (entry == null) continue;
            var entrySkillId = ReadStringValue(entry, new[] { "SkillId", "m_SkillId", "skillId", "m_skillId" }, "");
            if (!string.Equals(entrySkillId, skillId, StringComparison.Ordinal))
            {
                continue;
            }
            var guids = ReadMemberValue(entry, new[] { "ValidTargetActorGuids", "m_ValidTargetActorGuids" }) as System.Collections.IEnumerable;
            if (guids == null) continue;
            foreach (var guidRaw in guids)
            {
                try
                {
                    var guid = Convert.ToUInt32(guidRaw);
                    if (guid != 0) return guid;
                }
                catch { }
            }
        }
        return 0;
    }

    private static SkillPlan BuildSkillPlan(object probeRoot, object actorInstance, object? actorController, PendingAction action)
    {
        var skillIds = ReadStringListFromMethod(actorInstance, "GetEquippedCombatSkillIds");
        var selectedSkill = action.SkillIdx.HasValue && action.SkillIdx.Value >= 0 && action.SkillIdx.Value < skillIds.Count
            ? skillIds[action.SkillIdx.Value]
            : null;
        var targetGuid = ResolveTargetActorGuid(probeRoot, action.TargetIdx ?? 0, action.TargetTeam);
        bool? isValidSkill = null;
        bool? isValidTarget = null;
        if (actorController != null && !string.IsNullOrWhiteSpace(selectedSkill))
        {
            isValidSkill = InvokeBoolMethod(actorController, "GetIsValidSkill", selectedSkill);
            isValidTarget = targetGuid != 0
                ? InvokeBoolMethod(actorController, "GetIsValidSkillTarget", selectedSkill, targetGuid)
                : null;
        }
        return new SkillPlan(skillIds, selectedSkill, targetGuid, isValidSkill, isValidTarget);
    }

    private static SkillPlan BuildItemPlan(object probeRoot, object actorInstance, object? actorController, PendingAction action)
    {
        var itemIds = ReadStringListFromMethod(actorInstance, "GetCombatSkillInventoryIds");
        var selectedItem = !string.IsNullOrWhiteSpace(action.ItemId)
            ? action.ItemId
            : itemIds.FirstOrDefault();
        var targetGuid = ResolveTargetActorGuid(probeRoot, action.TargetIdx ?? 0, action.TargetTeam);
        bool? isValidSkill = null;
        bool? isValidTarget = null;
        if (actorController != null && !string.IsNullOrWhiteSpace(selectedItem))
        {
            RecalculateSkillTargets(actorController, selectedItem);
            isValidSkill = InvokeBoolMethod(actorController, "GetIsValidSkill", selectedItem);
            isValidTarget = targetGuid != 0
                ? InvokeBoolMethod(actorController, "GetIsValidSkillTarget", selectedItem, targetGuid)
                : null;
        }
        return new SkillPlan(itemIds, selectedItem, targetGuid, isValidSkill, isValidTarget);
    }

    private static void RecalculateSkillTargets(object actorController, string skillId)
    {
        try
        {
            InvokeOptional(actorController, "ClearValidSkillTargetEntries", true);
            var ids = new List<string> { skillId };
            var calculated = InvokeBoolMethod(actorController, "CalculateValidSkillTargetEntries", ids, true);
            DebugLog.Info($"RecalculateSkillTargets skill_id={skillId} calculated={FormatNullableBool(calculated)}");
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"RecalculateSkillTargets failed: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private static bool TryExecuteSelectedSkillPath(object probeRoot, object actorInstance, object? actorController, PendingAction action, SkillPlan plan, string preSig)
    {
        if (string.IsNullOrWhiteSpace(plan.SkillId))
        {
            DebugLog.Warn("Selected-skill path skipped: skill_id missing.");
            return false;
        }
        if (plan.TargetGuid == 0)
        {
            DebugLog.Warn("Selected-skill path skipped: target_guid missing.");
            return false;
        }
        if (plan.IsValidSkill == false || plan.IsValidTarget == false)
        {
            // GetIsValidSkill is unreliable under BepInEx (smart-enum reflection issue):
            // the game's UI shows the skill as enabled but our reflection cache reads false.
            // Proceed with the commit anyway - the actual game commit will produce no state delta
            // if the skill is truly invalid, and the agent will ban the action via the normal path.
            DebugLog.Info($"Selected-skill path: validity probe negative skill={FormatNullableBool(plan.IsValidSkill)} target={FormatNullableBool(plan.IsValidTarget)}; attempting commit anyway.");
        }

        try
        {
            if (!string.IsNullOrWhiteSpace(action.ItemId))
            {
                if (actorController != null)
                {
                    RecalculateSkillTargets(actorController, plan.SkillId);
                }
                TryWriteMemberValue(actorInstance, new[] { "m_selectedCombatItemId", "SelectedCombatItemId", "selectedCombatItemId" }, plan.SkillId);
                InvokeVoidByName(actorInstance, "SetSelectedCombatItem", plan.SkillId);
            }
            if (!InvokeVoidByName(actorInstance, "SetSelectedSkill", plan.SkillId))
            {
                DebugLog.Warn("Selected-skill path failed: SetSelectedSkill unavailable.");
                return false;
            }
            if (!InvokeVoidByName(actorInstance, "SetSelectedTargetActorGuid", plan.TargetGuid))
            {
                DebugLog.Warn("Selected-skill path failed: SetSelectedTargetActorGuid unavailable.");
                return false;
            }

            DebugLog.Info($"Selected-skill actor state after setters: {RenderSelectedSkillState(actorInstance)}");

            var afterSelectSig = TryBuildStateSignature(probeRoot);
            DebugLog.Info($"Selected-skill path prepared skill_id={plan.SkillId} target_guid={plan.TargetGuid} delta={HasStateDelta(preSig, afterSelectSig)} pre={preSig} post={afterSelectSig}");

            if (!string.IsNullOrWhiteSpace(action.ItemId) && TryExecuteUiItemPath(probeRoot, action, plan, preSig))
            {
                return true;
            }

            if (string.IsNullOrWhiteSpace(action.ItemId) && TryExecuteUiSkillPath(probeRoot, action, plan, preSig))
            {
                return true;
            }

            if (actorController == null)
            {
                DebugLog.Warn("Selected-skill path failed: actor controller missing for commit.");
                return false;
            }

            if (TryCommitSelectedSkillWithControllerInput(actorController, action, afterSelectSig, preSig, probeRoot))
            {
                return true;
            }

            var lastSig = afterSelectSig;
            var onTurnSelect = actorController.GetType().GetMethod("OnTurnSelect", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (onTurnSelect != null)
            {
                InvokeWithProbe(actorController, onTurnSelect, action, "selected_skill.OnTurnSelect", true, lastSig, out lastSig, probeRoot);
            }
            var onEndTurn = actorController.GetType().GetMethod("OnEndTurn", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (onEndTurn != null)
            {
                InvokeWithProbe(actorController, onEndTurn, action, "selected_skill.OnEndTurn", true, lastSig, out lastSig, probeRoot);
            }
            return HasStateDelta(preSig, lastSig);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Selected-skill path failed: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static bool TryCommitSelectedSkillWithControllerInput(object actorController, PendingAction action, string afterSelectSig, string preSig, object probeRoot)
    {
        var t = actorController.GetType();
        if ((t.FullName ?? string.Empty).IndexOf("ActorControllerInput", StringComparison.OrdinalIgnoreCase) < 0)
        {
            return false;
        }

        var selectSkill = t.GetMethod("AttemptSelectSkill", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var attemptTarget = t.GetMethod("AttemptSelectTarget", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var onTurnSelect = t.GetMethod("OnTurnSelect", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var onEndTurn = t.GetMethod("OnEndTurn", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (selectSkill == null && attemptTarget == null && onTurnSelect == null && onEndTurn == null)
        {
            return false;
        }

        try
        {
            var lastSig = afterSelectSig;
            if (selectSkill != null)
            {
                InvokeWithProbe(actorController, selectSkill, action, "selected_skill.AttemptSelectSkill", true, lastSig, out lastSig, probeRoot);
                if (HasStateDelta(preSig, lastSig)) return true;
            }
            if (attemptTarget != null)
            {
                InvokeWithProbe(actorController, attemptTarget, action, "selected_skill.AttemptSelectTarget", true, lastSig, out lastSig, probeRoot);
                if (HasStateDelta(preSig, lastSig)) return true;
            }
            if (onTurnSelect != null)
            {
                InvokeWithProbe(actorController, onTurnSelect, action, "selected_skill.OnTurnSelect", true, lastSig, out lastSig, probeRoot);
                if (HasStateDelta(preSig, lastSig)) return true;
            }
            if (onEndTurn != null)
            {
                InvokeWithProbe(actorController, onEndTurn, action, "selected_skill.OnEndTurn", true, lastSig, out lastSig, probeRoot);
                if (HasStateDelta(preSig, lastSig)) return true;
            }

            DebugLog.Info("Selected-skill controller-input commit executed without state delta.");
            return false;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Selected-skill controller-input commit failed: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static bool TryExecuteUiSkillPath(object probeRoot, PendingAction action, SkillPlan plan, string preSig)
    {
        var skillSelectionType = ClassPaths.ResolveType("Assets.Code.UI.SkillSelectionBhv");
        if (skillSelectionType == null)
        {
            DebugLog.Warn("UI skill path skipped: SkillSelectionBhv type missing.");
            return false;
        }

        object? skillSelection = null;
        try
        {
            if (typeof(UnityEngine.Object).IsAssignableFrom(skillSelectionType))
            {
                skillSelection = UnityEngine.Object.FindObjectOfType(skillSelectionType);
            }
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"UI skill path lookup failed: {ex.Message}");
        }
        if (skillSelection == null)
        {
            DebugLog.Warn("UI skill path skipped: SkillSelectionBhv instance missing.");
            return false;
        }

        var preferSkillId = string.Equals(plan.SkillId, "pass_stress", StringComparison.Ordinal);
        object? button = null;
        if (button == null && action.SkillIdx.HasValue)
        {
            button = InvokeObjectByName(skillSelection, "GetSkillButton", action.SkillIdx.Value)
                ?? InvokeObjectByName(skillSelection, "GetSkillButtonWithSkillButtonIndex", action.SkillIdx.Value);
        }
        if (button == null && !preferSkillId && !string.IsNullOrWhiteSpace(plan.SkillId))
        {
            button = InvokeObjectByName(skillSelection, "GetSkillButton", plan.SkillId);
        }
        if (button == null)
        {
            DebugLog.Warn($"UI skill path skipped: no skill button for skill_idx={action.SkillIdx?.ToString() ?? "null"} skill_id={plan.SkillId ?? "null"}.");
            return false;
        }

        DebugLog.Info($"UI skill button invoke: target={button.GetType().FullName} skill_idx={action.SkillIdx?.ToString() ?? "null"} skill_id={plan.SkillId ?? "null"}");
        InvokeOptional(button, "OnClick", true);
        InvokeOptional(button, "Submit");

        var afterButtonSig = TryBuildStateSignature(probeRoot);
        DebugLog.Info($"UI skill button post delta={HasStateDelta(preSig, afterButtonSig)} pre={preSig} post={afterButtonSig}");

        var hitbox = FindCombatActorHitbox(plan.TargetGuid);
        if (hitbox == null)
        {
            DebugLog.Warn($"UI skill path skipped: target hitbox missing for target_guid={plan.TargetGuid}.");
            return false;
        }

        DebugLog.Info($"UI target hitbox invoke: target={hitbox.GetType().FullName} target_guid={plan.TargetGuid}");
        var clicked = InvokeOptional(hitbox, "OnClick");

        var afterTargetSig = TryBuildStateSignature(probeRoot);
        DebugLog.Info($"UI target hitbox post delta={HasStateDelta(preSig, afterTargetSig)} pre={preSig} post={afterTargetSig}");
        if (HasStateDelta(preSig, afterTargetSig))
        {
            return true;
        }

        if (clicked)
        {
            DebugLog.Info("UI target hitbox click accepted without immediate state delta; treating as async skill commit.");
            return true;
        }

        return false;
    }

    private static bool TryExecuteUiItemPath(object probeRoot, PendingAction action, SkillPlan plan, string preSig)
    {
        if (string.IsNullOrWhiteSpace(plan.SkillId))
        {
            return false;
        }

        var candidates = new[]
        {
            "Assets.Code.UI.Items.CombatInventoryItemBhv",
            "Assets.Code.UI.Items.InventoryItemBhv",
            "Assets.Code.UI.Items.PlayerInventoryItemBhv"
        };

        object? itemBhv = null;
        foreach (var typeName in candidates)
        {
            var type = ClassPaths.ResolveType(typeName);
            if (type == null) continue;
            foreach (var obj in FindUnityObjects(type, includeInactive: true))
            {
                var rendered = RenderObjectMembers(obj, maxMembers: 20);
                DebugLog.Info($"UI item candidate: {rendered}");
                if (
                    rendered.IndexOf(plan.SkillId, StringComparison.OrdinalIgnoreCase) >= 0
                    || ObjectGraphContainsString(obj, plan.SkillId, depth: 2, visited: new HashSet<object>())
                )
                {
                    itemBhv = obj;
                    break;
                }
            }
            if (itemBhv != null) break;
        }

        if (itemBhv == null)
        {
            DebugLog.Warn($"UI item path skipped: no combat item UI object for item_id={plan.SkillId}.");
            return false;
        }

        DebugLog.Info($"UI item invoke: target={itemBhv.GetType().FullName} item_id={plan.SkillId}");
        InvokeOptional(itemBhv, "OnClick", true);
        InvokeOptional(itemBhv, "OnClick");
        InvokeOptional(itemBhv, "Submit");
        InvokeOptional(itemBhv, "OnSubmit");

        var afterItemSig = TryBuildStateSignature(probeRoot);
        DebugLog.Info($"UI item post delta={HasStateDelta(preSig, afterItemSig)} pre={preSig} post={afterItemSig}");

        var hitbox = FindCombatActorHitbox(plan.TargetGuid);
        if (hitbox == null)
        {
            DebugLog.Warn($"UI item path skipped: target hitbox missing for target_guid={plan.TargetGuid}.");
            return false;
        }

        DebugLog.Info($"UI item target hitbox invoke: target={hitbox.GetType().FullName} target_guid={plan.TargetGuid}");
        var clicked = InvokeOptional(hitbox, "OnClick");

        var afterTargetSig = TryBuildStateSignature(probeRoot);
        DebugLog.Info($"UI item target post delta={HasStateDelta(preSig, afterTargetSig)} pre={preSig} post={afterTargetSig}");
        if (HasStateDelta(preSig, afterTargetSig))
        {
            return true;
        }

        if (clicked)
        {
            DebugLog.Info("UI item target click accepted without immediate state delta; treating as async item commit.");
            return true;
        }

        return false;
    }

    private static object? FindCombatActorHitbox(uint actorGuid)
    {
        var hitboxType = ClassPaths.ResolveType("Assets.Code.UI.CombatActorHitboxBhv");
        if (hitboxType == null || !typeof(UnityEngine.Object).IsAssignableFrom(hitboxType))
        {
            return null;
        }

        UnityEngine.Object[] hitboxes;
        try
        {
            hitboxes = UnityEngine.Object.FindObjectsOfType(hitboxType);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"UI hitbox lookup failed: {ex.Message}");
            return null;
        }

        foreach (var hitbox in hitboxes)
        {
            if (hitbox == null) continue;
            var actorBhv = ReadMemberValue(hitbox, new[] { "m_CachedActorBhv", "ActorBhv", "m_ActorBhv" });
            var actorInstance = ReadMemberValue(actorBhv ?? new object(), new[] { "ActorInstance", "Actor", "m_Actor", "m_ActorInstance" });
            var guid = ReadUIntValue(actorInstance ?? actorBhv ?? hitbox, new[] { "ActorGuid", "Guid", "m_ActorGuid", "ActorDataGuid", "DataGuid" }, 0);
            DebugLog.Info($"UI hitbox candidate: {hitbox.GetType().FullName} guid={guid} actor_bhv={actorBhv?.GetType().FullName ?? "null"}");
            if (guid == actorGuid)
            {
                return hitbox;
            }
        }
        return null;
    }

    private static object? FindCombatActorBhv(uint actorGuid)
    {
        var type = ClassPaths.ResolveType("Assets.Code.Combat.CombatActorBhv");
        if (type == null)
        {
            return null;
        }
        foreach (var obj in FindUnityObjects(type, includeInactive: true))
        {
            var guid = ReadUIntValue(obj, new[] { "ActorGuid", "m_ActorGuid" }, 0);
            if (guid == actorGuid)
            {
                return obj;
            }
        }
        return null;
    }

    private static object? InvokeObjectByName(object target, string methodName, params object[] args)
    {
        var method = target.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == args.Length);
        if (method == null) return null;
        try
        {
            return method.Invoke(target, args);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"{methodName} UI invoke failed: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static List<object> FindUnityObjects(Type objectType, bool includeInactive)
    {
        var found = new List<object>();
        if (!typeof(UnityEngine.Object).IsAssignableFrom(objectType))
        {
            return found;
        }

        try
        {
            if (includeInactive)
            {
                var resourcesMethod = typeof(UnityEngine.Resources).GetMethod(
                    "FindObjectsOfTypeAll",
                    BindingFlags.Public | BindingFlags.Static,
                    null,
                    new[] { typeof(Type) },
                    null);
                if (resourcesMethod != null)
                {
                    var all = resourcesMethod.Invoke(null, new object[] { objectType }) as System.Collections.IEnumerable;
                    if (all != null)
                    {
                        foreach (var obj in all)
                        {
                            if (obj != null) found.Add(obj);
                        }
                        if (found.Count > 0) return found;
                    }
                }
            }

            var objects = UnityEngine.Object.FindObjectsOfType(objectType);
            foreach (var obj in objects)
            {
                if (obj != null) found.Add(obj);
            }
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Unity object lookup failed for {objectType.FullName}: {ex.Message}");
        }
        return found;
    }

    private static bool InvokeOptional(object target, string methodName, params object?[] args)
    {
        var method = target.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == args.Length);
        if (method == null) return false;
        try
        {
            method.Invoke(target, args);
            return true;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"{methodName} optional invoke failed: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static bool InvokeVoidByName(object target, string methodName, params object[] args)
    {
        var method = target.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == args.Length);
        if (method == null) return false;
        method.Invoke(target, args);
        return true;
    }

    private static string RenderSelectedSkillState(object actorInstance)
    {
        var selectedSkill = ReadStringValue(actorInstance, new[] { "SelectedSkillId", "SelectSkilIId", "SelectedSkill", "m_SelectedSkillId", "m_SelectSkillId" }, "unknown");
        var selectedTarget = ReadUIntValue(actorInstance, new[] { "SelectedTargetActorGuid", "TargetActorGuid", "m_SelectedTargetActorGuid", "m_TargetActorGuid" }, 0);
        return $"selected_skill={selectedSkill} selected_target_guid={selectedTarget}";
    }

    private static void LogSkillDiagnostics(SkillPlan plan, object actorInstance, object? actorController, PendingAction action)
    {
        try
        {
            DebugLog.Info(
                $"Skill diagnostics: actor={actorInstance.GetType().FullName} skill_idx={action.SkillIdx?.ToString() ?? "null"} " +
                $"skill_id={plan.SkillId ?? "null"} equipped=[{string.Join(",", plan.EquippedSkillIds)}] target_idx={action.TargetIdx?.ToString() ?? "null"} target_guid={plan.TargetGuid}");

            if (actorController != null && !string.IsNullOrWhiteSpace(plan.SkillId))
            {
                DebugLog.Info(
                    $"Skill diagnostics validity: skill_id={plan.SkillId} is_valid_skill={FormatNullableBool(plan.IsValidSkill)} " +
                    $"target_guid={plan.TargetGuid} is_valid_target={FormatNullableBool(plan.IsValidTarget)}");
            }

            if (actorController != null)
            {
                var entries = InvokeEnumerableMethod(actorController, "GetValidSkillTargetEntries");
                if (entries != null)
                {
                    var rendered = new List<string>();
                    foreach (var entry in entries)
                    {
                        if (entry == null) continue;
                        rendered.Add(RenderObjectMembers(entry, maxMembers: 8));
                        if (rendered.Count >= 12) break;
                    }
                    DebugLog.Info($"Skill diagnostics valid target entries: {string.Join(" || ", rendered)}");
                }
            }
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"Skill diagnostics failed: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private static string FormatNullableBool(bool? value)
    {
        return value.HasValue ? value.Value.ToString() : "unknown";
    }

    private static List<string> ReadStringListFromMethod(object target, string methodName)
    {
        var method = target.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == 0);
        if (method == null) return new List<string>();
        var result = method.Invoke(target, Array.Empty<object?>());
        if (result is not System.Collections.IEnumerable enumerable) return new List<string>();
        var values = new List<string>();
        foreach (var item in enumerable)
        {
            if (item != null) values.Add(item.ToString() ?? string.Empty);
        }
        return values.Where(v => !string.IsNullOrWhiteSpace(v)).ToList();
    }

    private static bool? InvokeBoolMethod(object target, string methodName, params object[] args)
    {
        var method = target.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == args.Length);
        if (method == null) return null;
        try
        {
            var result = method.Invoke(target, args);
            return result is bool b ? b : null;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"{methodName} diagnostic invoke failed: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static System.Collections.IEnumerable? InvokeEnumerableMethod(object target, string methodName)
    {
        var method = target.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == 0);
        if (method == null) return null;
        try
        {
            return method.Invoke(target, Array.Empty<object?>()) as System.Collections.IEnumerable;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"{methodName} diagnostic invoke failed: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static uint ResolveTargetActorGuid(object probeRoot, int targetIdx, string? targetTeam = null)
    {
        var battle = ExtractBattleRoot(probeRoot);
        var teams = ReadMemberValue(battle, new[] { "m_BattleTeams", "BattleTeams", "Teams" });
        var getTeam = teams?.GetType().GetMethod("GetTeam", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (getTeam == null) return 0;

        var preferAllyFirst = string.Equals(targetTeam, "heroes", StringComparison.OrdinalIgnoreCase);
        var teamOrder = preferAllyFirst ? new[] { 0, 1 } : new[] { 1, 0 };

        foreach (var teamIndex in teamOrder)
        {
            var team = getTeam.Invoke(teams, new object[] { teamIndex });
            if (team == null) continue;
            var actors = ReadMemberValue(team, new[] { "Actors", "m_Actors", "Units" }) as System.Collections.IEnumerable;
            if (actors == null) continue;

            // First pass: prefer matching by Slot/TeamPosition equal to targetIdx (matches snapshot semantics).
            foreach (var actor in actors)
            {
                if (actor == null) continue;
                var slot = ReadIntValue(actor, new[] { "TeamPosition", "Slot", "Index", "Position" }, -1);
                if (slot == targetIdx)
                {
                    var guid = ReadUIntValue(actor, new[] { "ActorGuid", "Guid", "m_ActorGuid", "ActorDataGuid", "DataGuid" }, 0);
                    if (guid != 0) return guid;
                }
            }

            // Fallback: positional match.
            var idx = 0;
            foreach (var actor in actors)
            {
                if (actor == null) { continue; }
                if (idx == targetIdx)
                {
                    var guid = ReadUIntValue(actor, new[] { "ActorGuid", "Guid", "m_ActorGuid", "ActorDataGuid", "DataGuid" }, 0);
                    if (guid != 0) return guid;
                }
                idx++;
            }

            // If a specific team was requested, do not fall through to the other team.
            if (targetTeam != null) break;
        }
        return 0;
    }

    private static string RenderObjectMembers(object obj, int maxMembers)
    {
        var parts = new List<string>();
        var type = obj.GetType();
        foreach (var prop in type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance).Take(maxMembers))
        {
            try { parts.Add($"{prop.Name}={prop.GetValue(obj)}"); } catch { }
        }
        foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance).Take(maxMembers - parts.Count))
        {
            try { parts.Add($"{field.Name}={field.GetValue(obj)}"); } catch { }
        }
        return $"{type.Name}{{{string.Join(",", parts)}}}";
    }

    private static bool ObjectGraphContainsString(object? obj, string needle, int depth, HashSet<object> visited)
    {
        if (obj == null || depth < 0 || string.IsNullOrWhiteSpace(needle)) return false;
        if (obj is string s) return s.IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0;
        if (obj.GetType().IsValueType) return obj.ToString()?.IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0;
        if (!visited.Add(obj)) return false;

        var direct = obj.ToString();
        if (!string.IsNullOrWhiteSpace(direct) && direct.IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return true;
        }

        var type = obj.GetType();
        foreach (var prop in type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance).Take(24))
        {
            if (prop.GetIndexParameters().Length != 0) continue;
            object? value = null;
            try { value = prop.GetValue(obj); } catch { }
            if (ObjectGraphContainsString(value, needle, depth - 1, visited)) return true;
        }
        foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance).Take(24))
        {
            object? value = null;
            try { value = field.GetValue(obj); } catch { }
            if (ObjectGraphContainsString(value, needle, depth - 1, visited)) return true;
        }
        return false;
    }

    private static bool TryExecutePassButtonSequence(PendingAction action, bool commitProbeMode, string preSig, object probeRoot, uint actorGuid)
    {
        var buttonType = ClassPaths.PassTurnButtonCandidates
            .Select(name => ClassPaths.ResolveType(name))
            .FirstOrDefault(t => t != null);
        if (buttonType == null)
        {
            DebugLog.Warn("Pass button type not found.");
            return false;
        }

        var buttons = FindUnityObjects(buttonType, includeInactive: true);
        var button = buttons.FirstOrDefault();

        if (button == null)
        {
            DebugLog.Warn($"Pass button instance missing: {buttonType.FullName}");
            return false;
        }

        if (actorGuid != 0)
        {
            InvokeOptional(button, "OnSelectedActor", actorGuid);
        }

        var methods = buttonType
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Where(m => !m.IsAbstract && !m.IsSpecialName)
            .Where(m => ClassPaths.MethodPassButtonCandidates.Any(name => string.Equals(m.Name, name, StringComparison.Ordinal)))
            .Where(m => m.GetParameters().Length == 0)
            .OrderByDescending(ScorePassButtonMethod)
            .ToList();

        if (methods.Count == 0)
        {
            var hints = buttonType
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .Where(m => !m.IsSpecialName)
                .Select(DescribeMethod)
                .Take(60);
            DebugLog.Warn($"Pass button has no zero-arg candidate methods. Methods: {string.Join(" || ", hints)}");
            return false;
        }

        var lastSig = preSig;
        var invokedAny = false;
        foreach (var method in methods)
        {
            DebugLog.Info($"Pass button invoke: target={button.GetType().FullName} method={DescribeMethod(method)}");
            if (!InvokeWithProbe(button, method, action, $"pass_button.{method.Name}", commitProbeMode, lastSig, out lastSig, probeRoot))
            {
                continue;
            }
            invokedAny = true;
            if (HasStateDelta(preSig, lastSig))
            {
                return true;
            }
        }

        if (invokedAny)
        {
            DebugLog.Info("Pass button invoke accepted without immediate state delta; treating as async pass commit.");
            return true;
        }

        DebugLog.Warn("Pass button candidates invoked but no state delta observed.");
        return false;
    }

    private static int ScorePassButtonMethod(MethodInfo method)
    {
        var name = method.Name;
        if (name.IndexOf("Click", StringComparison.OrdinalIgnoreCase) >= 0) return 100;
        if (name.IndexOf("Pass", StringComparison.OrdinalIgnoreCase) >= 0) return 90;
        if (name.IndexOf("EndTurn", StringComparison.OrdinalIgnoreCase) >= 0) return 80;
        return 0;
    }

    private static void LogControllerMethodHints(Type type)
    {
        var hints = type
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Where(m =>
                m.Name.IndexOf("Skill", StringComparison.OrdinalIgnoreCase) >= 0 ||
                m.Name.IndexOf("Target", StringComparison.OrdinalIgnoreCase) >= 0 ||
                m.Name.IndexOf("Turn", StringComparison.OrdinalIgnoreCase) >= 0 ||
                m.Name.IndexOf("Confirm", StringComparison.OrdinalIgnoreCase) >= 0 ||
                m.Name.IndexOf("End", StringComparison.OrdinalIgnoreCase) >= 0)
            .Select(DescribeMethod)
            .Distinct()
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .Take(80)
            .ToList();

        DebugLog.Info($"ActorController method hints ({type.FullName}): {string.Join(" || ", hints)}");
    }

    private static bool InvokeWithProbe(object target, MethodInfo method, PendingAction action, string step, bool commitProbeMode, string preSig, out string postSig, object? probeRoot = null)
    {
        try
        {
            method.Invoke(target, BuildArgs(method, action));
            postSig = TryBuildStateSignature(probeRoot ?? target);
            DebugLog.Info($"Commit probe {step} ok delta={HasStateDelta(preSig, postSig)} pre={preSig} post={postSig}");
            return true;
        }
        catch (Exception ex)
        {
            postSig = preSig;
            DebugLog.Warn($"Commit probe {step} failed: {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    private static bool HasStateDelta(string before, string after)
    {
        return !string.Equals(before, after, StringComparison.Ordinal);
    }

    private static bool IsBattleOver(object controller)
    {
        var value = ReadMemberValue(controller, new[] { "IsBattleOver", "Done", "BattleEnded", "m_IsBattleOver" });
        if (value == null) return false;
        try { return Convert.ToBoolean(value); } catch { return false; }
    }

    private static string TryBuildStateSignature(object root)
    {
        try
        {
            var controller = ExtractBattleRoot(root);
            if (ReadMemberValue(controller, new[] { "m_BattleTeams", "BattleTeams", "Teams" }) == null && CombatHooks.CurrentTurnContext != null)
            {
                controller = ExtractBattleRoot(CombatHooks.CurrentTurnContext);
            }
            var active = ResolveCurrentActor(controller);
            var activeSide = ReadIntValue(active ?? controller, new[] { "TeamIndex" }, -1);
            var activeIdx = ReadIntValue(active ?? controller, new[] { "TeamPosition", "Slot", "Index", "Position" }, -1);
            var round = ReadIntValue(controller, new[] { "CurrentRound", "m_CurrentRound", "Round", "RoundNumber" }, -1);
            var done = IsBattleOver(controller);
            var heroesHp = ReadTeamHpSignature(controller, 0);
            var enemiesHp = ReadTeamHpSignature(controller, 1);
            return $"r={round};a={activeSide}:{activeIdx};d={done};h={heroesHp};e={enemiesHp}";
        }
        catch
        {
            return "sig_unavailable";
        }
    }

    private static object ExtractBattleRoot(object controller)
    {
        var battle = ReadMemberValue(controller, new[] { "m_Battle", "Battle" });
        return battle ?? controller;
    }

    private static object? ResolveBestBattleRoot(object fallback)
    {
        var combatType = ClassPaths.ResolveType("Assets.Code.Combat.CombatBhv");
        if (combatType != null && typeof(UnityEngine.Object).IsAssignableFrom(combatType))
        {
            try
            {
                var combat = UnityEngine.Object.FindObjectOfType(combatType);
                if (combat != null)
                {
                    var battle = ReadMemberValue(combat, new[] { "m_Battle", "Battle" });
                    if (battle != null) return battle;
                    return combat;
                }
            }
            catch (Exception ex)
            {
                DebugLog.Warn($"Resolve CombatBhv battle root failed: {ex.Message}");
            }
        }

        var extracted = ExtractBattleRoot(fallback);
        return extracted ?? fallback;
    }

    private static string ReadTeamHpSignature(object controller, int teamIndex)
    {
        var teams = ReadMemberValue(controller, new[] { "m_BattleTeams", "BattleTeams", "Teams" });
        if (teams == null) return "-";
        var getTeam = teams.GetType().GetMethod("GetTeam", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var team = getTeam?.Invoke(teams, new object[] { teamIndex });
        if (team == null) return "-";
        var actors = ReadMemberValue(team, new[] { "Actors", "m_Actors", "Units" }) as System.Collections.IEnumerable;
        if (actors == null) return "-";

        var sb = new StringBuilder();
        var slot = 0;
        foreach (var actor in actors)
        {
            if (actor == null) continue;
            var hp = ReadIntValue(actor, new[] { "DisplayedHp", "HpRounded", "CurrentHP", "HP", "Health" }, 0);
            var alive = ReadBoolValue(actor, new[] { "IsLiving", "IsAlive", "Alive" }, hp > 0) ? 1 : 0;
            if (sb.Length > 0) sb.Append(',');
            sb.Append(slot).Append(':').Append(hp).Append(':').Append(alive);
            slot++;
        }
        return sb.ToString();
    }

    private static object? ReadMemberValue(object target, string[] names)
    {
        foreach (var name in names)
        {
            var type = target.GetType();
            var prop = type.GetProperty(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (prop != null) return prop.GetValue(target);
            var field = type.GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (field != null) return field.GetValue(target);
        }
        return null;
    }

    private static int ReadIntValue(object target, string[] names, int fallback)
    {
        var value = ReadMemberValue(target, names);
        if (value == null) return fallback;
        try { return Convert.ToInt32(value); } catch { return fallback; }
    }

    private static uint ReadUIntValue(object target, string[] names, uint fallback)
    {
        var value = ReadMemberValue(target, names);
        if (value == null) return fallback;
        try { return Convert.ToUInt32(value); } catch { return fallback; }
    }

    private static string ReadStringValue(object target, string[] names, string fallback)
    {
        var value = ReadMemberValue(target, names);
        return value?.ToString() ?? fallback;
    }

    private static bool TryWriteMemberValue(object target, string[] names, object value)
    {
        foreach (var name in names)
        {
            var type = target.GetType();
            var field = type.GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (field != null)
            {
                field.SetValue(target, Convert.ChangeType(value, field.FieldType));
                return true;
            }
            var prop = type.GetProperty(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (prop != null && prop.CanWrite)
            {
                prop.SetValue(target, Convert.ChangeType(value, prop.PropertyType), null);
                return true;
            }
        }
        return false;
    }

    private static bool ReadBoolValue(object target, string[] names, bool fallback)
    {
        var value = ReadMemberValue(target, names);
        if (value == null) return fallback;
        try { return Convert.ToBoolean(value); } catch { return fallback; }
    }

    private readonly record struct PendingAction(
        int RequestId,
        int HeroSlot,
        int? SkillIdx,
        int? TargetIdx,
        string? ItemId,
        bool PassTurn,
        string? TargetTeam = null,
        int? MoveDelta = null);

    private readonly record struct SkillPlan(
        List<string> EquippedSkillIds,
        string? SkillId,
        uint TargetGuid,
        bool? IsValidSkill,
        bool? IsValidTarget);
}
