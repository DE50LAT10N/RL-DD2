// BepInEx plugin bootstrap for DDRL.
// Initializes config, logging, IPC server, state reader, dispatcher, and combat hooks.
// Entry point loaded by DD2 through BepInEx.

using System;
using System.Collections.Generic;
using System.IO;
using BepInEx;
using DdRL.Plugin.Actions;
using DdRL.Plugin.Config;
using DdRL.Plugin.Hooks;
using DdRL.Plugin.Ipc;
using DdRL.Plugin.Logging;
using DdRL.Plugin.State;

namespace DdRL.Plugin;

[BepInPlugin("com.rl.ddrl", "DDRL", "0.1.21")]
public sealed class Plugin : BaseUnityPlugin
{
    private static readonly string BootstrapLogPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "DDRL",
        "mono_bootstrap.log");

    private JsonLineServer? _server;
    private StateReader? _reader;
    private Dispatcher? _dispatcher;
    private RuntimeConfig? _cfg;
    private float _nextPumpAt;
    private float _suppressStatePumpUntil;
    private const float PumpIntervalSeconds = 0.25f;
    private const float BattleEndQuietSeconds = 5.0f;

    private void Awake()
    {
        TraceBootstrap("Awake entered");
        try
        {
            TraceBootstrap("Loading config");
            _cfg = RuntimeConfig.Load(Config);
            TraceBootstrap("Initializing debug logger");
            DebugLog.Init(Logger);
            DebugLog.Info("DDRL plugin loading");
            DebugLog.Info($"Runtime config: host={_cfg.Host} port={_cfg.Port} commit_probe={_cfg.CommitProbeMode} skill_hooks={_cfg.EnableSkillHooks} discovery_dump={_cfg.EnableDiscoveryDump} data_dump={_cfg.EnableDataDump}");
            TraceBootstrap("DDRL plugin loading logged");

            if (_cfg.EnableDiscoveryDump)
            {
                TraceBootstrap("Starting class discovery");
                ClassDiscovery.Discover(dumpToDisk: true);
                TraceBootstrap("Class discovery complete");
            }

            if (_cfg.EnableDataDump)
            {
                TraceBootstrap("Starting data dump");
                DataExtractor.DumpAllScriptableObjects();
                TraceBootstrap("Data dump complete");
                if (_cfg.DumpOnlyExit)
                {
                    DebugLog.Info("DumpOnlyExit enabled, terminating process after data dump.");
                    TraceBootstrap("DumpOnlyExit triggered");
                    Environment.Exit(0);
                    return;
                }
            }

            TraceBootstrap("Starting JSON line server");
            _server = new JsonLineServer(_cfg.Host, _cfg.Port);
            _server.OnMessage = HandleClientMessage;
            _server.OnClientConnected = () => _server.Send(Protocol.MakeHello("unknown", inBattle: false));
            _server.Start();
            TraceBootstrap("JSON line server started");

            _dispatcher = new Dispatcher(_server, _cfg.CommitProbeMode, _cfg.EnableSkillHooks);
            _reader = new StateReader(_server);
            CombatHooks.Install(
                onTurnBegin: () => _reader.PokeAndPublish(),
                onProcessPending: () => _dispatcher.ProcessPending(),
                onBattleEnd: won =>
                {
                    _suppressStatePumpUntil = UnityEngine.Time.unscaledTime + BattleEndQuietSeconds;
                    _server.Send(Protocol.MakeBattleEnd(won));
                }
            );

            _server.Send(Protocol.MakeHello("unknown", inBattle: false));
            DebugLog.Info($"DDRL plugin ready on {_cfg.Host}:{_cfg.Port}");
            TraceBootstrap("Plugin ready");
        }
        catch (Exception ex)
        {
            Logger.LogError($"DDRL init failed: {ex}");
            TraceBootstrap($"Init failed: {ex}");
        }
    }

    private void Update()
    {
        if (_dispatcher == null) return;
        if (UnityEngine.Time.unscaledTime < _nextPumpAt) return;

        _nextPumpAt = UnityEngine.Time.unscaledTime + PumpIntervalSeconds;

        if (_reader != null && UnityEngine.Time.unscaledTime >= _suppressStatePumpUntil)
        {
            try { _reader.PokeAndPublish(); }
            catch (Exception ex) { DebugLog.Warn($"Periodic state pump failed: {ex.Message}"); }
        }
        else if (_reader == null)
        {
            DebugLog.Warn("StateReader is null in Update; skipping state pump.");
        }

        if (_dispatcher.PendingCount > 0)
        {
            DebugLog.Info($"Update sees pending actions: {_dispatcher.PendingCount}");
        }

        try { _dispatcher.ProcessPending(); }
        catch (Exception ex) { DebugLog.Warn($"Periodic dispatcher pump failed: {ex.Message}"); }
    }

    private static void TraceBootstrap(string message)
    {
        try
        {
            var dir = Path.GetDirectoryName(BootstrapLogPath);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }

            var line = $"[{DateTime.UtcNow:O}] {message}{Environment.NewLine}";
            File.AppendAllText(BootstrapLogPath, line);
        }
        catch
        {
            // Swallow logging failures to avoid affecting plugin startup.
        }
    }

    private void HandleClientMessage(Dictionary<string, object?> msg)
    {
        var type = ReadString(msg, "type");
        if (string.Equals(type, "ping", StringComparison.OrdinalIgnoreCase))
        {
            _server?.Send(Protocol.MakePong(ReadInt(msg, "request_id")));
            if (UnityEngine.Time.unscaledTime >= _suppressStatePumpUntil)
            {
                try { _reader?.PokeAndPublish(force: true); }
                catch (Exception ex) { DebugLog.Warn($"Ping-triggered state publish failed: {ex.Message}"); }
            }
            return;
        }
        if (string.Equals(type, "action", StringComparison.OrdinalIgnoreCase))
        {
            DebugLog.Info("Incoming action message received.");
            HandleAction(msg);
            return;
        }
        _server?.Send(Protocol.MakeError("unknown_type", $"unknown message type: {type}"));
    }

    private void HandleAction(Dictionary<string, object?> msg)
    {
        if (_dispatcher == null || _server == null) return;

        var requestId = ReadInt(msg, "request_id");
        try
        {
            var heroSlot = ReadInt(msg, "hero_slot");
            var skillIdx = TryReadInt(msg, "skill_idx");
            var targetIdx = TryReadInt(msg, "target_idx");
            var moveDelta = TryReadInt(msg, "move_delta");
            var itemId = TryReadString(msg, "item_id");
            // For a move action, accept move_skill_id from the client and pass it via ItemId
            // (Dispatcher.TryExecuteMove uses ItemId as the transport for move_skill_id).
            if (moveDelta.HasValue && string.IsNullOrWhiteSpace(itemId))
            {
                itemId = TryReadString(msg, "move_skill_id");
            }
            var passTurn = ReadBool(msg, "pass_turn", false);
            var targetTeam = TryReadString(msg, "target_team");
            DebugLog.Info($"Queue action request_id={requestId} pass_turn={passTurn} hero_slot={heroSlot} skill_idx={skillIdx?.ToString() ?? "null"} target_idx={targetIdx?.ToString() ?? "null"} move_delta={moveDelta?.ToString() ?? "null"} target_team={targetTeam ?? "null"} item_id={itemId ?? "null"}");
            _dispatcher.EnqueueAction(requestId, heroSlot, skillIdx, targetIdx, itemId, passTurn, targetTeam, moveDelta);
            // HandleAction is drained on Unity main thread (see DrainMainThreadActions),
            // so we can process immediately and return ack in the same frame.
            _dispatcher.ProcessPending();
        }
        catch (Exception ex)
        {
            _server.Send(Protocol.MakeAck(requestId, ok: false, reason: $"bad_payload: {ex.Message}"));
        }
    }

    private static int ReadInt(Dictionary<string, object?> msg, string key)
    {
        if (!msg.TryGetValue(key, out var value) || value == null) return 0;
        if (value is long l) return (int)l;
        if (value is int i) return i;
        if (value is double d) return (int)d;
        return int.TryParse(value.ToString(), out var parsed) ? parsed : 0;
    }

    private static int? TryReadInt(Dictionary<string, object?> msg, string key)
    {
        if (!msg.TryGetValue(key, out var value) || value == null) return null;
        if (value is long l) return (int)l;
        if (value is int i) return i;
        if (value is double d) return (int)d;
        return int.TryParse(value.ToString(), out var parsed) ? parsed : null;
    }

    private static bool ReadBool(Dictionary<string, object?> msg, string key, bool fallback)
    {
        if (!msg.TryGetValue(key, out var value) || value == null) return fallback;
        if (value is bool b) return b;
        return bool.TryParse(value.ToString(), out var parsed) ? parsed : fallback;
    }

    private static string ReadString(Dictionary<string, object?> msg, string key, string fallback = "")
    {
        if (!msg.TryGetValue(key, out var value) || value == null) return fallback;
        return value.ToString() ?? fallback;
    }

    private static string? TryReadString(Dictionary<string, object?> msg, string key)
    {
        if (!msg.TryGetValue(key, out var value) || value == null) return null;
        return value.ToString();
    }
}
