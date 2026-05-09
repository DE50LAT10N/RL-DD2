// Runtime configuration for the DDRL BepInEx plugin.
// Reads host, port, hook, debug, and dump options from BepInEx config.
// Used during plugin startup before IPC and hooks are installed.

using BepInEx.Configuration;

namespace DdRL.Plugin.Config;

public sealed class RuntimeConfig
{
    public string Host { get; init; } = "127.0.0.1";
    public int Port { get; init; } = 8765;
    public bool EnableDiscoveryDump { get; init; } = false;
    public bool EnableDataDump { get; init; } = false;
    public bool DumpOnlyExit { get; init; } = false;
    public bool EnableHarmonyFileLog { get; init; } = false;
    public bool CommitProbeMode { get; init; } = false;
    public bool EnableSkillHooks { get; init; } = true;

    public static RuntimeConfig Load(ConfigFile config)
    {
        var host = config.Bind("Network", "Host", "127.0.0.1", "TCP host for NDJSON server.").Value;
        var port = config.Bind("Network", "Port", 8765, "TCP port for NDJSON server.").Value;
        var discovery = config.Bind("Debug", "EnableDiscoveryDump", false, "Dump discovered classes to %APPDATA%/DDRL.").Value;
        var dataDump = config.Bind("Debug", "EnableDataDump", false, "Dump DD2 ScriptableObjects to %APPDATA%/DDRL/data.").Value;
        var dumpOnly = config.Bind("Debug", "DumpOnlyExit", false, "Dump data and exit process (for repeatable dumps).").Value;
        var harmonyLog = config.Bind("Debug", "EnableHarmonyFileLog", false, "Enable Harmony file log.").Value;
        var commitProbe = config.Bind("Debug", "CommitProbeMode", false, "Probe commit path and require observed battle state delta before ack ok=true.").Value;
        var enableSkillHooks = config.Bind("Experimental", "EnableSkillHooks", true, "Enable experimental live skill execution hooks. May crash DD2.").Value;

        return new RuntimeConfig
        {
            Host = host,
            Port = port,
            EnableDiscoveryDump = discovery,
            EnableDataDump = dataDump,
            DumpOnlyExit = dumpOnly,
            EnableHarmonyFileLog = harmonyLog,
            CommitProbeMode = commitProbe,
            EnableSkillHooks = enableSkillHooks
        };
    }
}
