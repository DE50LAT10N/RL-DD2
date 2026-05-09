// TCP JSON-lines server used by the live Python agent.
// Sends hello/state/ack messages and receives action/ping requests.
// Normalizes Newtonsoft JToken values before handing them to plugin logic.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using DdRL.Plugin.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace DdRL.Plugin.Ipc;

public sealed class JsonLineServer : IDisposable
{
    private readonly string _host;
    private readonly int _port;
    private readonly JsonSerializerSettings _jsonSettings = new()
    {
        ContractResolver = new DefaultContractResolver
        {
            NamingStrategy = new CamelCaseNamingStrategy()
        }
    };
    private readonly ConcurrentQueue<string> _sendQueue = new();

    private TcpListener? _listener;
    private TcpClient? _client;
    private CancellationTokenSource? _cts;
    private Task? _acceptTask;
    private Task? _readTask;
    private Task? _writeTask;

    public Action<Dictionary<string, object?>>? OnMessage { get; set; }
    public Action? OnClientConnected { get; set; }

    public JsonLineServer(string host, int port)
    {
        _host = host;
        _port = port;
    }

    public void Start()
    {
        if (_listener != null) return;
        _cts = new CancellationTokenSource();
        _listener = new TcpListener(IPAddress.Parse(_host), _port);
        _listener.Start();
        _acceptTask = Task.Run(() => AcceptLoop(_cts.Token));
        DebugLog.Info($"NDJSON server listening on {_host}:{_port}");
    }

    public void Stop()
    {
        var cts = _cts;
        if (cts == null) return;
        cts.Cancel();
        CloseClient();
        try { _listener?.Stop(); } catch { }
        _listener = null;
        _cts = null;
    }

    public bool IsClientConnected => _client?.Connected ?? false;

    public void Send(Dictionary<string, object?> payload)
    {
        var line = JsonConvert.SerializeObject(payload, _jsonSettings) + "\n";
        _sendQueue.Enqueue(line);
    }

    private async Task AcceptLoop(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                var accepted = await _listener!.AcceptTcpClientAsync();
                CloseClient();
                _client = accepted;
                var stream = accepted.GetStream();
                _readTask = Task.Run(() => ReadLoop(stream, token), token);
                _writeTask = Task.Run(() => WriteLoop(stream, token), token);
                DebugLog.Info("NDJSON client connected");
                OnClientConnected?.Invoke();
            }
            catch (Exception ex) when (!token.IsCancellationRequested)
            {
                DebugLog.Warn($"Accept loop error: {ex.Message}");
                await Task.Delay(200, token);
            }
        }
    }

    private async Task ReadLoop(NetworkStream stream, CancellationToken token)
    {
        using var reader = new StreamReader(stream);
        while (!token.IsCancellationRequested && _client?.Connected == true)
        {
            string? line;
            try
            {
                line = await reader.ReadLineAsync();
            }
            catch (Exception ex)
            {
                DebugLog.Warn($"Read loop failed: {ex.Message}");
                break;
            }

            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            try
            {
                var msg = JsonConvert.DeserializeObject<Dictionary<string, object?>>(line!);
                if (msg == null) continue;
                var normalized = Normalize(msg);
                OnMessage?.Invoke(normalized);
            }
            catch (Exception ex)
            {
                DebugLog.Warn($"Failed to parse incoming line: {ex.Message}");
            }
        }
        CloseClient();
    }

    private async Task WriteLoop(NetworkStream stream, CancellationToken token)
    {
        using var writer = new StreamWriter(stream) { AutoFlush = true };
        while (!token.IsCancellationRequested && _client?.Connected == true)
        {
            if (!_sendQueue.TryDequeue(out var line))
            {
                await Task.Delay(5, token);
                continue;
            }
            try
            {
                await writer.WriteAsync(line);
            }
            catch (Exception ex)
            {
                DebugLog.Warn($"Write loop failed: {ex.Message}");
                break;
            }
        }
        CloseClient();
    }

    private static Dictionary<string, object?> Normalize(Dictionary<string, object?> msg)
    {
        // Newtonsoft.Json returns JToken; normalize the top level to
        // primitive CLR types so the rest of the plugin can consume it.
        var result = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        foreach (var kv in msg)
        {
            result[kv.Key] = ConvertJsonValue(kv.Value);
        }
        return result;
    }

    private static object? ConvertJsonValue(object? value)
    {
        if (value is JToken token)
        {
            return token.Type switch
            {
                JTokenType.String => token.Value<string>(),
                JTokenType.Integer => token.Value<long>(),
                JTokenType.Float => token.Value<double>(),
                JTokenType.Boolean => token.Value<bool>(),
                JTokenType.Null => null,
                JTokenType.Array => token,
                JTokenType.Object => token,
                _ => token.ToString()
            };
        }
        return value;
    }

    private void CloseClient()
    {
        try { _client?.Close(); } catch { }
        _client = null;
    }

    public void Dispose() => Stop();
}
