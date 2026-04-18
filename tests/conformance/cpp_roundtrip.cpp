// Minimal round-trip tool using the C++ FieldML-API.
// Usage: cpp_roundtrip <input.fieldml> <output.fieldml>
// Exit 0 on success, non-zero on failure.

#include <cstdio>
#include <cstring>

#include "fieldml_api.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <input.fieldml> <output.fieldml>\n", argv[0]);
        return 2;
    }

    FmlSessionHandle session = Fieldml_CreateFromFile(argv[1]);
    if (session == FML_INVALID_HANDLE) {
        std::fprintf(stderr, "cpp_roundtrip: failed to load %s\n", argv[1]);
        return 1;
    }

    int status = Fieldml_WriteFile(session, argv[2]);
    Fieldml_Destroy(session);

    if (status != 0) {
        std::fprintf(stderr, "cpp_roundtrip: Fieldml_WriteFile returned %d\n", status);
        return 1;
    }
    return 0;
}
