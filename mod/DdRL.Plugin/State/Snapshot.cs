// Serializable live battle snapshot model.
// Converts plugin-read unit/action state into JSON dictionaries for Python.
// Includes unit stats, tokens, deaths-door/death-armor, and precomputed legal actions.

using System;
using System.Collections.Generic;
using System.Linq;

namespace DdRL.Plugin.State;

public sealed record UnitToken(string Id, int Count);

public sealed record UnitSnapshot(
    string Name,
    int Slot,
    bool Alive,
    int Hp,
    int MaxHp,
    int Rank,
    int Stress,
    int Speed,
    int Size,
    List<UnitToken> Tokens);

public sealed record Snapshot(
    bool InBattle,
    string Phase,
    int Round,
    string ActiveSide,
    int ActiveIndex,
    List<UnitSnapshot> Heroes,
    List<UnitSnapshot> Enemies,
    bool Done,
    bool? HeroesWon,
    List<Dictionary<string, object?>>? PrecomputedLegalActions = null)
{
    public Dictionary<string, object?> ToMessage()
    {
        return new Dictionary<string, object?>
        {
            ["type"] = "state",
            ["in_battle"] = InBattle,
            ["phase"] = Phase,
            ["round"] = Round,
            ["active_unit"] = new Dictionary<string, object?>
            {
                ["side"] = ActiveSide,
                ["index"] = ActiveIndex
            },
            ["heroes"] = Heroes.Select(UnitToDict).ToList(),
            ["enemies"] = Enemies.Select(UnitToDict).ToList(),
            ["legal_actions"] = PrecomputedLegalActions ?? BuildLegalActions(),
            ["done"] = Done,
            ["heroes_won"] = HeroesWon
        };
    }

    private List<Dictionary<string, object?>> BuildLegalActions()
    {
        var actions = new List<Dictionary<string, object?>>();
        if (Done || !InBattle)
        {
            return actions;
        }

        if (string.Equals(ActiveSide, "heroes", StringComparison.OrdinalIgnoreCase))
        {
            var activeHero = Heroes.FirstOrDefault(h => h.Slot == ActiveIndex && h.Alive);
            if (activeHero != null)
            {
                foreach (var enemy in Enemies.Where(e => e.Alive))
                {
                    for (var skillIdx = 0; skillIdx < 4; skillIdx++)
                    {
                        actions.Add(new Dictionary<string, object?>
                        {
                            ["hero_slot"] = activeHero.Slot,
                            ["skill_idx"] = skillIdx,
                            ["target_idx"] = enemy.Slot
                        });
                    }
                }
                actions.Add(new Dictionary<string, object?>
                {
                    ["hero_slot"] = activeHero.Slot,
                    ["move_delta"] = -1
                });
                actions.Add(new Dictionary<string, object?>
                {
                    ["hero_slot"] = activeHero.Slot,
                    ["move_delta"] = 1
                });
            }
        }
        return actions;
    }

    private static Dictionary<string, object?> UnitToDict(UnitSnapshot unit)
    {
        return new Dictionary<string, object?>
        {
            ["name"] = unit.Name,
            ["slot"] = unit.Slot,
            ["alive"] = unit.Alive,
            ["hp"] = unit.Hp,
            ["max_hp"] = unit.MaxHp,
            ["rank"] = unit.Rank,
            ["stress"] = unit.Stress,
            ["speed"] = unit.Speed,
            ["size"] = unit.Size,
            ["death_door"] = unit.Tokens.Any(t => t.Id.IndexOf("death", StringComparison.OrdinalIgnoreCase) >= 0 && t.Id.IndexOf("door", StringComparison.OrdinalIgnoreCase) >= 0),
            ["death_armor"] = unit.Tokens.Where(t => t.Id.IndexOf("death", StringComparison.OrdinalIgnoreCase) >= 0 && t.Id.IndexOf("armor", StringComparison.OrdinalIgnoreCase) >= 0).Sum(t => t.Count),
            ["tokens"] = unit.Tokens.Select(t => new Dictionary<string, object?>
            {
                ["id"] = t.Id,
                ["count"] = t.Count
            }).ToList()
        };
    }
}
