// Compatibility shim for C# record types on older target frameworks.
// Lets the plugin use modern record syntax while targeting netstandard2.1.
// No runtime behavior beyond compiler support.

namespace System.Runtime.CompilerServices;

internal static class IsExternalInit
{
}
