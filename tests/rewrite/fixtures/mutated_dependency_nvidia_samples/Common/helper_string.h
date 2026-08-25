#ifndef COMMON_HELPER_STRING_H_
#define COMMON_HELPER_STRING_H_

inline void ascifyTestTelemetry() {}
inline bool checkCmdLineFlag(int, const char**, const char*) {
  ascifyTestTelemetry();
  return false;
}
inline int getCmdLineArgumentInt(int, const char**, const char*) { return 0; }

#endif
