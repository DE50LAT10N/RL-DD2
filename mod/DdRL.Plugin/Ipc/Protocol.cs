using System;
using System.Collections.Generic;
using DdRL.Plugin.State;

namespace DdRL.Plugin.Ipc;

public static class Protocol
{
    public const int ProtocolVersion = 1;
    public const string ModVersion = "0.1.22-move-skill-id";

    public static Dictionary<string, object?> MakeHello(string gameVersion, bool inBattle)
    {
        return new Dictionary<string, object?>
        {
            ["type"] = "hello",
            ["game"] = "dd2",
            ["mod_version"] = ModVersion,
            ["protocol_version"] = ProtocolVersion,
            ["game_version"] = gameVersion,
            ["in_battle"] = inBattle
        };
    }

    public static Dictionary<string, object?> MakeState(Snapshot snapshot)
    {
        var msg = snapshot.ToMessage();
        if (!msg.ContainsKey("relationships")) msg["relationships"] = new List<object>();
        if (!msg.ContainsKey("items_available")) msg["items_available"] = new Dictionary<string, int>();
        return msg;
    }

    public static Dictionary<string, object?> MakeAck(int requestId, bool ok, string? reason = null, string? method = null)
    {
        var j = new Dictionary<string, object?>
        {
            ["type"] = "ack",
            ["request_id"] = requestId,
            ["ok"] = ok
        };
        if (!string.IsNullOrWhiteSpace(reason)) j["reason"] = reason;
        if (!string.IsNullOrWhiteSpace(method)) j["method"] = method;
        return j;
    }

    public static Dictionary<string, object?> MakePong(int requestId)
    {
        return new Dictionary<string, object?> { ["type"] = "pong", ["request_id"] = requestId };
    }

    public static Dictionary<string, object?> MakeBattleEnd(bool heroesWon)
    {
        return new Dictionary<string, object?> { ["type"] = "battle_end", ["heroes_won"] = heroesWon };
    }

    public static Dictionary<string, object?> MakeError(string code, string message)
    {
        return new Dictionary<string, object?> { ["type"] = "error", ["code"] = code, ["message"] = message };
    }
}
