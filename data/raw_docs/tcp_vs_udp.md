# TCP vs UDP (Quick Notes)
TCP is connection-oriented and provides reliable, ordered delivery with congestion control. It is suitable when correctness matters (e.g., file transfer, web requests).
UDP is connectionless and sends datagrams without built-in reliability or ordering. It is often used for real-time or loss-tolerant applications (e.g., VoIP, gaming, telemetry).
Trade-offs: TCP reliability & ordering vs UDP lower overhead and potentially lower latency.
