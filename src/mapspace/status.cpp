#include "pytimeloop/mapspace/status.h"

namespace pytimeloop::pymapspace {

std::string StatusRepr(mapspace::Status& s) {
  std::stringstream ss;
  ss << "MappingConstructionStatus(success=";
  if (s.success) {
    ss << "True";
  } else {
    ss << "False";
  }
  ss << ", '" << s.fail_reason << "')";
  return ss.str();
}

}  // namespace pytimeloop::pymapspace
