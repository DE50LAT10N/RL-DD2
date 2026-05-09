// Live DD2 battle state reader.
// Reflects combat teams, active unit, stats, tokens, and authoritative legal actions from the game.
// Feeds Snapshot messages to the JSON-lines IPC server.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using DdRL.Plugin.Hooks;
using DdRL.Plugin.Ipc;
using DdRL.Plugin.Logging;

namespace DdRL.Plugin.State;

public sealed class StateReader
{
    private readonly JsonLineServer _server;
    private Snapshot? _last;

    public StateReader(JsonLineServer server)
    {
        _server = server;
    }

    public void PokeAndPublish(bool force = false)
    {
        var next = ReadSnapshot();
        if (!force && _last != null && _last.Equals(next)) return;
        _last = next;
        _server.Send(Protocol.MakeState(next));
    }

    private static Snapshot ReadSnapshot()
    {
        var controllerType = ClassPaths.ResolveType(ClassPaths.BattleControllerCandidates);
        if (controllerType != null)
        {
            var controller = ResolveInstance(controllerType);
            if (controller != null)
            {
                controller = ExtractBattleRoot(controller);
                try
                {
                    var heroes = ReadUnits(controller, new[] { "m_BattleTeams", "BattleTeams", "Teams" }, isHero: true);
                    var enemies = ReadUnits(controller, new[] { "m_BattleTeams", "BattleTeams", "Teams" }, isHero: false);
                    if (heroes.Count > 0 || enemies.Count > 0)
                    {
                        var round = ReadInt(controller, new[] { "CurrentRound", "m_CurrentRound", "Round", "RoundNumber" }, 0);
                        var active = ResolveActive(controller);
                        var done = ReadBool(controller, new[] { "IsBattleOver", "Done", "BattleEnded", "m_IsBattleOver" }, false);
                        bool? heroesWon = done ? ResolveHeroesWon(controller, heroes, enemies) : null;
                        var validLegalActions = done ? null : BuildValidLegalActions(controller, active.side, active.index, enemies);
                        return new Snapshot(true, active.side == "heroes" ? "hero_turn" : "enemy_turn", round, active.side, active.index, heroes, enemies, done, heroesWon, validLegalActions);
                    }
                }
                catch (Exception ex)
                {
                    DebugLog.Warn($"State read from controller failed: {ex.Message}");
                }
            }
        }

        var turnContext = CombatHooks.CurrentTurnContext;
        if (turnContext != null)
        {
            var fromTurnContext = ReadSnapshotFromTurnContext(turnContext);
            if (fromTurnContext != null) return fromTurnContext;
        }

        return new Snapshot(false, "transition", 0, "heroes", 0, new List<UnitSnapshot>(), new List<UnitSnapshot>(), false, null);
    }

    private static bool ResolveHeroesWon(object controller, List<UnitSnapshot> heroes, List<UnitSnapshot> enemies)
    {
        foreach (var name in new[] { "HeroesWon", "PlayerWon", "DidHeroesWin", "IsHeroVictory", "IsPartyVictorious" })
        {
            var value = ReadMember(controller, new[] { name });
            if (value == null) continue;
            try { return Convert.ToBoolean(value); } catch { }
        }

        var heroesAlive = heroes.Any(h => h.Alive);
        var enemiesAlive = enemies.Any(e => e.Alive);
        if (!enemiesAlive && heroesAlive) return true;
        return false;
    }

    private static Snapshot? ReadSnapshotFromTurnContext(object turnContext)
    {
        var teams = ReadMember(turnContext, new[] { "m_BattleTeams", "BattleTeams", "Teams" });
        if (teams == null) return null;

        var heroes = ReadUnitsFromTeams(teams, 0, true);
        var enemies = ReadUnitsFromTeams(teams, 1, false);
        if (heroes.Count == 0 && enemies.Count == 0) return null;

        var active = ReadMember(turnContext, new[] { "m_PerformerActor", "PerformerActor", "CurrentActor" });
        var side = ReadInt(active ?? turnContext, new[] { "TeamIndex" }, 0) == 0 ? "heroes" : "enemies";
        var index = ReadInt(active ?? turnContext, new[] { "TeamPosition", "Slot", "Index", "Position" }, 0);
        var round = ReadInt(turnContext, new[] { "m_CurrentRound", "CurrentRound", "Round" }, 0);
        return new Snapshot(true, side == "heroes" ? "hero_turn" : "enemy_turn", round, side, index, heroes, enemies, false, null);
    }

    private static object? ResolveInstance(Type controllerType)
    {
        var prop = controllerType.GetProperty("Instance", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.FlattenHierarchy);
        if (prop != null) return prop.GetValue(null);
        var field = controllerType.GetField("Instance", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.FlattenHierarchy);
        if (field != null) return field.GetValue(null);
        if (typeof(UnityEngine.Object).IsAssignableFrom(controllerType))
        {
            return UnityEngine.Object.FindObjectOfType(controllerType);
        }
        return null;
    }

    private static object ExtractBattleRoot(object controller)
    {
        var battle = ReadMember(controller, new[] { "m_Battle", "Battle" });
        return battle ?? controller;
    }

    private static List<UnitSnapshot> ReadUnits(object controller, string[] collectionNames, bool isHero)
    {
        var collection = ResolveTeamCollection(controller, collectionNames, isHero ? 0 : 1);
        if (collection is not IEnumerable enumerable) return new List<UnitSnapshot>();

        var list = new List<UnitSnapshot>();
        var slot = 0;
        foreach (var u in enumerable)
        {
            if (u == null) continue;
            var name = ReadString(u, new[] { "ActorDataId", "ActorName", "Id", "Name", "UnitName" }, isHero ? $"hero_{slot}" : $"enemy_{slot}");
            var hp = ReadInt(u, new[] { "DisplayedHp", "HpRounded", "CurrentHP", "HP", "Health" }, 0);
            var maxHp = ReadInt(u, new[] { "DisplayedHpMax", "CurrentHpMax", "MaxHP", "MaxHealth" }, hp);
            var rank = ReadInt(u, new[] { "TeamPosition", "Rank", "CombatRank", "Position" }, slot + 1);
            var stress = ReadInt(u, new[] { "Stress", "CurrentStress", "DisplayedStress", "m_Stress" }, 0);
            var speed = ReadInt(u, new[] { "Speed", "CombatSpeed", "m_Speed" }, 0);
            var size = ReadInt(u, new[] { "Size", "CombatSize", "m_Size" }, 1);
            var alive = ReadBool(u, new[] { "IsLiving", "IsAlive", "Alive" }, hp > 0);
            var tokens = ReadTokens(u);
            list.Add(new UnitSnapshot(name, slot, alive, hp, maxHp, rank, stress, speed, size, tokens));
            slot++;
        }
        return list;
    }

    private static List<UnitSnapshot> ReadUnitsFromTeams(object teams, int teamIndex, bool isHero)
    {
        var getTeam = teams.GetType().GetMethod("GetTeam", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var team = getTeam?.Invoke(teams, new object[] { teamIndex });
        if (team == null) return new List<UnitSnapshot>();

        var actors = ReadMember(team, new[] { "Actors", "m_Actors", "Units" });
        if (actors is not IEnumerable enumerable) return new List<UnitSnapshot>();

        var list = new List<UnitSnapshot>();
        var slot = 0;
        foreach (var u in enumerable)
        {
            if (u == null) continue;
            var name = ReadString(u, new[] { "ActorDataId", "ActorName", "Id", "Name", "UnitName" }, isHero ? $"hero_{slot}" : $"enemy_{slot}");
            var hp = ReadInt(u, new[] { "DisplayedHp", "HpRounded", "CurrentHP", "HP", "Health" }, 0);
            var maxHp = ReadInt(u, new[] { "DisplayedHpMax", "CurrentHpMax", "MaxHP", "MaxHealth" }, hp);
            var rank = ReadInt(u, new[] { "TeamPosition", "Rank", "CombatRank", "Position" }, slot + 1);
            var stress = ReadInt(u, new[] { "Stress", "CurrentStress", "DisplayedStress", "m_Stress" }, 0);
            var speed = ReadInt(u, new[] { "Speed", "CombatSpeed", "m_Speed" }, 0);
            var size = ReadInt(u, new[] { "Size", "CombatSize", "m_Size" }, 1);
            var alive = ReadBool(u, new[] { "IsLiving", "IsAlive", "Alive" }, hp > 0);
            var tokens = ReadTokens(u);
            list.Add(new UnitSnapshot(name, slot, alive, hp, maxHp, rank, stress, speed, size, tokens));
            slot++;
        }

        return list;
    }

    private static List<UnitToken> ReadTokens(object unit)
    {
        var tokenContainer = ReadMember(unit, new[] { "ReadOnlyTokenContainer", "TokenContainer", "Tokens", "CombatTokens" });
        var raw = tokenContainer ?? ReadMember(unit, new[] { "Tokens", "CombatTokens" });
        if (raw is not IEnumerable enumerable) return new List<UnitToken>();

        var tokens = new List<UnitToken>();
        foreach (var t in enumerable)
        {
            if (t == null) continue;
            var id = ReadString(t, new[] { "Id", "TokenId", "Name", "ActorDataId" }, "token");
            var count = ReadInt(t, new[] { "Count", "Stacks", "Value" }, 1);
            tokens.Add(new UnitToken(id, count));
        }
        return tokens;
    }

    private static List<Dictionary<string, object?>>? BuildValidLegalActions(object controller, string activeSide, int activeIndex, List<UnitSnapshot> enemies)
    {
        if (!string.Equals(activeSide, "heroes", StringComparison.OrdinalIgnoreCase)) return null;
        try
        {
            var activeActor = ResolveCurrentActor(controller);
            if (activeActor == null)
            {
                DebugLog.Info("BuildValidLegalActions skip: active actor unresolved.");
                return new List<Dictionary<string, object?>>();
            }

            var heroController = ReadMember(activeActor, new[] { "Controller", "m_ActorController" });
            if (heroController == null)
            {
                DebugLog.Info($"BuildValidLegalActions skip: hero controller missing on {activeActor.GetType().FullName}.");
                return new List<Dictionary<string, object?>>();
            }

            var equippedSkills = InvokeStringList(activeActor, "GetEquippedCombatSkillIds");
            var itemSkillIds = InvokeStringList(activeActor, "GetCombatSkillInventoryIds");

            var teams = ReadMember(controller, new[] { "m_BattleTeams", "BattleTeams", "Teams" });
            if (teams == null) return new List<Dictionary<string, object?>>();
            var getTeam = teams.GetType().GetMethod("GetTeam", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (getTeam == null) return new List<Dictionary<string, object?>>();

            var heroTargets = CollectTargets(getTeam, teams, 0);
            var enemyTargets = CollectTargets(getTeam, teams, 1);

            // Best-effort: try to populate the game's valid-skill cache. Fine to fail.
            PrepareValidSkillCalculations(heroController, equippedSkills);

            // Read the game's authoritative GetValidSkillTargetEntries() snapshot. This is the
            // ground truth the player UI uses. SkillType reflection is broken under BepInEx,
            // but this method works because it only returns SkillTargetEntry objects.
            var validEntries = ReadValidSkillTargetEntries(heroController);

            var actions = new List<Dictionary<string, object?>>();
            var actorRank = ReadInt(activeActor, new[] { "TeamPosition", "Slot", "Index", "Position" }, activeIndex);
            var emitted = 0;
            var emittedItems = 0;
            var emittedMoves = 0;
            string? moveSkillIdSeen = null;
            var sawPassEntry = false;

            // Index targets by guid for quick lookup.
            var heroByGuid = heroTargets.Where(t => t.guid != 0).ToDictionary(t => t.guid, t => t);
            var enemyByGuid = enemyTargets.Where(t => t.guid != 0).ToDictionary(t => t.guid, t => t);

            if (validEntries != null && validEntries.Count > 0)
            {
                foreach (var entry in validEntries)
                {
                    if (!entry.IsValid) continue;
                    var skillId = entry.SkillId ?? string.Empty;
                    if (string.IsNullOrWhiteSpace(skillId)) continue;

                    // Pass-like entries appear in the game's valid target list, but the
                    // current dispatcher cannot commit them reliably. Track for logs only.
                    if (skillId.IndexOf("pass", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        sawPassEntry = true;
                        continue;
                    }

                    // Move skill (id ends with "_move" or equals "move"). Convert each target into
                    // an explicit move_delta payload using the rank delta from the active actor.
                    var isMoveSkill = skillId.EndsWith("_move", StringComparison.OrdinalIgnoreCase)
                        || string.Equals(skillId, "move", StringComparison.OrdinalIgnoreCase);
                    if (isMoveSkill)
                    {
                        moveSkillIdSeen = skillId;
                        foreach (var guid in entry.ValidTargetGuids)
                        {
                            if (!heroByGuid.TryGetValue(guid, out var ally)) continue;
                            if (!ally.alive) continue;
                            var delta = ally.slot - actorRank;
                            if (delta == 0) continue;
                            // The generic DD2 combat move is a neighboring rank swap.
                            // Some target-entry probes include farther allies, but committing
                            // those through the move skill path often produces no state delta.
                            if (Math.Abs(delta) != 1) continue;
                            actions.Add(new Dictionary<string, object?>
                            {
                                ["hero_slot"] = activeIndex,
                                ["move_delta"] = delta,
                                ["move_skill_id"] = skillId,
                                ["target_idx"] = ally.slot,
                                ["target_team"] = "heroes",
                            });
                            emittedMoves++;
                        }
                        continue;
                    }

                    // Combat skill: try to map skill_id back to its equipped index.
                    var skillIdx = equippedSkills.FindIndex(s => string.Equals(s, skillId, StringComparison.Ordinal));
                    var isItem = skillIdx < 0 && itemSkillIds.Contains(skillId);

                    foreach (var guid in entry.ValidTargetGuids)
                    {
                        var (slot, team, alive) = ResolveSlotForGuid(guid, heroByGuid, enemyByGuid);
                        var inactiveEnemyRemnant = !alive
                            && string.Equals(team, "enemies", StringComparison.OrdinalIgnoreCase)
                            && IsEnemyRemnantTarget(enemies, slot);
                        if (slot < 0 || (!alive && !inactiveEnemyRemnant)) continue;
                        if (isItem)
                        {
                            actions.Add(new Dictionary<string, object?>
                            {
                                ["hero_slot"] = activeIndex,
                                ["item_id"] = skillId,
                                ["skill_id"] = skillId,
                                ["target_idx"] = slot,
                                ["target_team"] = team,
                            });
                            emittedItems++;
                        }
                        else if (skillIdx >= 0)
                        {
                            actions.Add(new Dictionary<string, object?>
                            {
                                ["hero_slot"] = activeIndex,
                                ["skill_idx"] = skillIdx,
                                ["skill_id"] = skillId,
                                ["target_idx"] = slot,
                                ["target_team"] = team,
                            });
                            emitted++;
                        }
                    }
                }
            }
            else
            {
                DebugLog.Info("BuildValidLegalActions: GetValidSkillTargetEntries returned empty; waiting for a valid hero action frame.");
            }

            DebugLog.Info($"BuildValidLegalActions emitted={emitted} emitted_items={emittedItems} emitted_moves={emittedMoves} entries={(validEntries?.Count ?? 0)} pass_entry_seen={sawPassEntry} move_skill_id={moveSkillIdSeen ?? "?"} actor_rank={actorRank} actions={actions.Count}.");
            return actions;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"BuildValidLegalActions failed: {ex.Message}");
            return new List<Dictionary<string, object?>>();
        }
    }

    private static bool IsEnemyRemnantTarget(List<UnitSnapshot> enemies, int slot)
    {
        var enemy = enemies.FirstOrDefault(e => e.Slot == slot);
        if (enemy == null) return false;
        var text = enemy.Name.ToLowerInvariant();
        return text.Contains("corpse")
            || text.Contains("cadaver")
            || text.Contains("remnant")
            || text.Contains("tomb")
            || text.Contains("grave")
            || text.Contains("gravestone")
            || text.Contains("headstone")
            || text.Contains("\u0442\u0440\u0443\u043f")
            || text.Contains("\u043d\u0430\u0434\u0433\u0440\u043e\u0431");
    }

    private readonly struct ValidSkillEntry
    {
        public ValidSkillEntry(string skillId, bool isValid, List<uint> validTargetGuids)
        {
            SkillId = skillId;
            IsValid = isValid;
            ValidTargetGuids = validTargetGuids;
        }
        public string SkillId { get; }
        public bool IsValid { get; }
        public List<uint> ValidTargetGuids { get; }
    }

    private static List<ValidSkillEntry>? ReadValidSkillTargetEntries(object heroController)
    {
        try
        {
            var method = heroController.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.FlattenHierarchy)
                .FirstOrDefault(m => string.Equals(m.Name, "GetValidSkillTargetEntries", StringComparison.Ordinal) && m.GetParameters().Length == 0);
            if (method == null) return null;
            var raw = method.Invoke(heroController, Array.Empty<object?>());
            if (raw is not IEnumerable enumerable) return null;
            var list = new List<ValidSkillEntry>();
            foreach (var entry in enumerable)
            {
                if (entry == null) continue;
                var skillId = ReadString(entry, new[] { "m_SkillId", "SkillId", "skillId", "m_skillId" }, "");
                var isValid = ReadBool(entry, new[] { "IsValid", "m_IsValid", "isValid" }, true);
                var guidsRaw = ReadMember(entry, new[] { "m_ValidTargetActorGuids", "ValidTargetActorGuids" }) as IEnumerable;
                var guids = new List<uint>();
                if (guidsRaw != null)
                {
                    foreach (var g in guidsRaw)
                    {
                        if (g == null) continue;
                        try { guids.Add(Convert.ToUInt32(g)); } catch { }
                    }
                }
                list.Add(new ValidSkillEntry(skillId, isValid, guids));
            }
            return list;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"ReadValidSkillTargetEntries failed: {ex.Message}");
            return null;
        }
    }

    private static (int slot, string team, bool alive) ResolveSlotForGuid(
        uint guid,
        Dictionary<uint, (int team, int slot, uint guid, bool alive)> heroByGuid,
        Dictionary<uint, (int team, int slot, uint guid, bool alive)> enemyByGuid)
    {
        if (heroByGuid.TryGetValue(guid, out var h)) return (h.slot, "heroes", h.alive);
        if (enemyByGuid.TryGetValue(guid, out var e)) return (e.slot, "enemies", e.alive);
        return (-1, "none", false);
    }

    private static List<(int team, int slot, uint guid, bool alive)> CollectTargets(MethodInfo getTeam, object teams, int teamIndex)
    {
        var team = getTeam.Invoke(teams, new object[] { teamIndex });
        if (team == null) return new List<(int, int, uint, bool)>();
        var actorsRaw = ReadMember(team, new[] { "Actors", "m_Actors", "Units" });
        if (actorsRaw is not IEnumerable actors) return new List<(int, int, uint, bool)>();

        var list = new List<(int team, int slot, uint guid, bool alive)>();
        foreach (var actor in actors)
        {
            if (actor == null) continue;
            var slot = ReadInt(actor, new[] { "TeamPosition", "Slot", "Index", "Position" }, list.Count);
            var guid = ReadUInt(actor, new[] { "ActorGuid", "Guid", "m_ActorGuid", "ActorDataGuid", "DataGuid" }, 0);
            var alive = ReadBool(actor, new[] { "IsLiving", "IsAlive", "Alive" }, true);
            list.Add((teamIndex, slot, guid, alive));
        }
        return list;
    }

    private static string? ResolveMoveSkillId(object actorInstance)
    {
        try
        {
            var baseSkills = InvokeStringList(actorInstance, "GetEquippedCombatSkillIds");
            var withMove = ReadEquippedSkillsIncludingMove(actorInstance, includeMoveSkill: true, includePassSkill: false);
            if (withMove != null && withMove.Count > 0)
            {
                var diff = withMove.Where(id => !baseSkills.Contains(id) && !string.IsNullOrWhiteSpace(id)).ToList();
                if (diff.Count > 0) return diff[0];
            }
            foreach (var name in new[] { "GetMoveSkillId", "GetCombatMoveSkillId", "MoveSkillId", "m_MoveSkillId" })
            {
                var member = ReadMember(actorInstance, new[] { name });
                if (member != null)
                {
                    var asString = member.ToString();
                    if (!string.IsNullOrWhiteSpace(asString)) return asString;
                }
            }
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"StateReader.ResolveMoveSkillId failed: {ex.Message}");
        }
        return null;
    }

    private static bool _loggedActorMethodHints;
    private static bool _loggedControllerMethodHints;

    /// <summary>
    /// Returns enum values for a type. Uses reflection-on-static-fields, which works even when
    /// <see cref="Type.IsEnum"/> erroneously reports false (a known BepInEx loader-context quirk
    /// that breaks both <see cref="Enum.GetValues"/> and <see cref="Enum.ToObject"/>).
    /// </summary>
    private static List<object> ResolveEnumValues(Type type)
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
            if (values.Count > 0) return values;
        }
        catch { }

        try
        {
            var enumValues = Enum.GetValues(type);
            foreach (var v in enumValues)
            {
                if (v != null) values.Add(v);
            }
            if (values.Count > 0) return values;
        }
        catch { }

        for (var i = 0; i < 16; i++)
        {
            try
            {
                var v = Enum.ToObject(type, i);
                if (v != null && !values.Contains(v)) values.Add(v);
            }
            catch { }
        }
        return values;
    }

    private static object? ResolveAnyEnumValue(Type type)
    {
        var values = ResolveEnumValues(type);
        return values.Count > 0 ? values[0] : null;
    }

    private static List<string>? ReadEquippedSkillsIncludingMove(object actorInstance, bool includeMoveSkill, bool includePassSkill)
    {
        try
        {
            var actorType = actorInstance.GetType();
            var allOverloads = actorType
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.FlattenHierarchy)
                .Where(m => string.Equals(m.Name, "GetEquippedCombatSkillIds", StringComparison.Ordinal))
                .ToList();

            if (!_loggedActorMethodHints)
            {
                _loggedActorMethodHints = true;
                var sigs = string.Join(" || ", allOverloads.Select(m =>
                    $"{m.Name}({string.Join(", ", m.GetParameters().Select(p => $"{p.ParameterType.FullName ?? p.ParameterType.Name} {p.Name}"))})"));
                DebugLog.Info($"GetEquippedCombatSkillIds overloads on {actorType.FullName}: {sigs}");
            }

            // Lenient match: 3 params where param[1] and param[2] are bool. Param[0] type is treated as the SkillType enum.
            var method = allOverloads.FirstOrDefault(m =>
            {
                var ps = m.GetParameters();
                return ps.Length == 3 && ps[1].ParameterType == typeof(bool) && ps[2].ParameterType == typeof(bool);
            });
            if (method == null)
            {
                DebugLog.Info("ReadEquippedSkillsIncludingMove: no 3-arg overload (?, bool, bool) found.");
                return null;
            }
            var firstParamType = method.GetParameters()[0].ParameterType;
            // Static-field enumeration works even when CLR thinks the type is not an enum
            // (a known BepInEx reflection quirk).
            var firstArgCandidates = new List<object?>();
            firstArgCandidates.AddRange(ResolveEnumValues(firstParamType));
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
                    DebugLog.Info($"GetEquippedCombatSkillIds 3-arg invoke failed for arg0={firstArg}: {ex.GetType().Name}: {ex.Message}");
                    continue;
                }
                if (raw is not IEnumerable enumerable) continue;
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
            DebugLog.Warn($"StateReader.ReadEquippedSkillsIncludingMove failed: {ex.Message}");
            return null;
        }
    }

    private static void PrepareValidSkillCalculations(object heroController, List<string> equippedSkills)
    {
        try
        {
            var controllerType = heroController.GetType();
            var allOverloads = controllerType
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.FlattenHierarchy)
                .Where(m => m.Name.IndexOf("CalculateValid", StringComparison.Ordinal) >= 0)
                .ToList();

            if (!_loggedControllerMethodHints)
            {
                _loggedControllerMethodHints = true;
                var sigs = string.Join(" || ", allOverloads.Select(m =>
                    $"{m.Name}({string.Join(", ", m.GetParameters().Select(p => $"{p.ParameterType.FullName ?? p.ParameterType.Name} {p.Name}"))})"));
                DebugLog.Info($"CalculateValid* overloads on {controllerType.FullName}: {sigs}");
            }

            var anyInvoked = false;

            // Preferred: SkillType-based overload calculates targets for entire skill class at once.
            foreach (var calcByType in allOverloads.Where(m =>
            {
                if (!string.Equals(m.Name, "CalculateValidCombatSkillTargetEntries", StringComparison.Ordinal)
                    && !string.Equals(m.Name, "CalculateValidSkillTargetEntries", StringComparison.Ordinal))
                    return false;
                var ps = m.GetParameters();
                if (ps.Length != 2) return false;
                if (ps[1].ParameterType != typeof(bool)) return false;
                if (ps[0].ParameterType == typeof(string)) return false;
                if (typeof(IEnumerable).IsAssignableFrom(ps[0].ParameterType)) return false;
                return true;
            }))
            {
                var enumType = calcByType.GetParameters()[0].ParameterType;
                var values = ResolveEnumValues(enumType);
                foreach (var enumValue in values)
                {
                    try
                    {
                        calcByType.Invoke(heroController, new object?[] { enumValue, true });
                        anyInvoked = true;
                    }
                    catch (Exception innerEx)
                    {
                        DebugLog.Info($"PrepareValidSkillCalculations({calcByType.Name}, {enumValue}) failed: {innerEx.GetType().Name}: {innerEx.Message}");
                    }
                }
            }

            // List-based overload as additional pass.
            if (equippedSkills.Count > 0)
            {
                var listOverload = allOverloads.FirstOrDefault(m =>
                {
                    if (!string.Equals(m.Name, "CalculateValidSkillTargetEntries", StringComparison.Ordinal)) return false;
                    var ps = m.GetParameters();
                    if (ps.Length != 2) return false;
                    if (ps[1].ParameterType != typeof(bool)) return false;
                    return typeof(IEnumerable).IsAssignableFrom(ps[0].ParameterType) && ps[0].ParameterType != typeof(string);
                });
                if (listOverload != null)
                {
                    try
                    {
                        listOverload.Invoke(heroController, new object?[] { equippedSkills, true });
                        anyInvoked = true;
                    }
                    catch (Exception innerEx)
                    {
                        DebugLog.Info($"PrepareValidSkillCalculations(list) failed: {innerEx.GetType().Name}: {innerEx.Message}");
                    }
                }
            }

            if (!anyInvoked)
            {
                DebugLog.Warn("PrepareValidSkillCalculations: no calc overload was invoked successfully.");
            }
            else
            {
                DebugLog.Info($"PrepareValidSkillCalculations: invoked OK on {controllerType.FullName}.");
            }
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"PrepareValidSkillCalculations failed: {ex.Message}");
        }
    }

    private static List<string> InvokeStringList(object target, string methodName)
    {
        try
        {
            var method = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == 0);
            if (method == null) return new List<string>();
            var result = method.Invoke(target, Array.Empty<object?>());
            if (result is not IEnumerable enumerable) return new List<string>();
            var values = new List<string>();
            foreach (var item in enumerable)
            {
                if (item == null) continue;
                values.Add(item.ToString() ?? string.Empty);
            }
            return values;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"InvokeStringList {methodName} failed: {ex.Message}");
            return new List<string>();
        }
    }

    private static bool? InvokeBool(object target, string methodName, params object[] args)
    {
        try
        {
            var method = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == args.Length);
            if (method == null) return null;
            var result = method.Invoke(target, args);
            return result is bool b ? b : null;
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"InvokeBool {methodName} failed: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static int? InvokeInt(object target, string methodName, params object[] args)
    {
        try
        {
            var method = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => string.Equals(m.Name, methodName, StringComparison.Ordinal) && m.GetParameters().Length == args.Length);
            if (method == null) return null;
            var result = method.Invoke(target, args);
            if (result == null) return null;
            return Convert.ToInt32(result);
        }
        catch (Exception ex)
        {
            DebugLog.Warn($"InvokeInt {methodName} failed: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static (string side, int index) ResolveActive(object controller)
    {
        var current = ResolveCurrentActor(controller);
        if (current == null) return ("heroes", 0);
        var side = ReadInt(current, new[] { "TeamIndex" }, 0) == 0 ? "heroes" : "enemies";
        var index = ReadInt(current, new[] { "TeamPosition", "Slot", "Index", "Position" }, 0);
        return (side, index);
    }

    private static object? ResolveCurrentActor(object controller)
    {
        var current = ReadMember(controller, new[] { "CurrentTurnOwner", "ActiveUnit", "CurrentUnit" });
        if (current != null) return current;
        var guid = ReadUInt(controller, new[] { "CurrentActorGuid", "m_CurrentActorGuid" }, 0);
        if (guid == 0) return null;
        var teams = ReadMember(controller, new[] { "m_BattleTeams", "BattleTeams", "Teams" });
        var getActor = teams?.GetType().GetMethod("GetActorFromActorGuid", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        return getActor?.Invoke(teams, new object[] { guid });
    }

    private static object? ResolveTeamCollection(object controller, IEnumerable<string> names, int teamIndex)
    {
        var teams = ReadMember(controller, names);
        if (teams == null) return null;

        var getTeam = teams.GetType().GetMethod("GetTeam", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var team = getTeam?.Invoke(teams, new object[] { teamIndex });
        if (team == null) return null;
        return ReadMember(team, new[] { "Actors", "m_Actors", "Units" });
    }

    private static object? ReadMember(object target, IEnumerable<string> names)
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

    private static int ReadInt(object target, IEnumerable<string> names, int fallback)
    {
        var value = ReadMember(target, names);
        if (value == null) return fallback;
        try { return Convert.ToInt32(value); } catch { return fallback; }
    }

    private static uint ReadUInt(object target, IEnumerable<string> names, uint fallback)
    {
        var value = ReadMember(target, names);
        if (value == null) return fallback;
        try { return Convert.ToUInt32(value); } catch { return fallback; }
    }

    private static bool ReadBool(object target, IEnumerable<string> names, bool fallback)
    {
        var value = ReadMember(target, names);
        if (value == null) return fallback;
        try { return Convert.ToBoolean(value); } catch { return fallback; }
    }

    private static string ReadString(object target, IEnumerable<string> names, string fallback)
    {
        var value = ReadMember(target, names);
        return value?.ToString() ?? fallback;
    }
}
