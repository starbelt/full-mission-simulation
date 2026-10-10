// Validate wire bytes BEFORE turning a received command into a C string.
#pragma once

namespace adcs {
inline bool commandByteValid(unsigned char c) {
    // ASCII command text plus permitted horizontal whitespace/CR/LF only.
    // In particular NUL must never truncate a command into a valid reset.
    return (c >= 0x20 && c <= 0x7e) || c == '\t' || c == '\r' || c == '\n';
}
// Preserve CR as whitespace, never erase it and join command/numeric tokens.
// The protocol's horizontal whitespace is space/tab; LF ends the wire line.
inline char commandTextByte(unsigned char c) {
    return c == '\r' ? ' ' : static_cast<char>(c);
}
}  // namespace adcs
