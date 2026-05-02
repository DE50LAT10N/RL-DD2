using BepInEx.Logging;

namespace DdRL.Plugin.Logging;

public static class DebugLog
{
    private static ManualLogSource? _log;

    public static void Init(ManualLogSource logSource) => _log = logSource;

    public static void Info(string message) => _log?.LogInfo(message);
    public static void Warn(string message) => _log?.LogWarning(message);
    public static void Error(string message) => _log?.LogError(message);
    public static void Debug(string message) => _log?.LogDebug(message);
}
