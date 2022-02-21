#pragma once

// Boost headers
#include <boost/multiprecision/cpp_int.hpp>

// Timeloop headers
#include "workload/util/per-data-space.hpp"

// PyBind11 headers
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

// Type casters
namespace pybind11 {
namespace detail {

namespace mp = boost::multiprecision;

/* Type caster for problem::PerDataSpace -> List */
template <typename Type>
struct type_caster<problem::PerDataSpace<Type>> {
  using value_conv = make_caster<Type>;

  bool load(handle src, bool convert) {
    if (!isinstance<sequence>(src) || isinstance<bytes>(src) ||
        isinstance<str>(src)) {
      return false;
    }
    auto l = reinterpret_borrow<sequence>(src);
    if (l.size() != problem::GetShape()->NumDataSpaces) {
      return false;
    }
    size_t ctr = 0;
    for (auto it : l) {
      value_conv conv;
      if (!conv.load(it, convert)) return false;
      value[ctr++] = cast_op<Type &&>(std::move(conv));
    }
    return true;
  }

  template <typename T>
  static handle cast(T &&src, return_value_policy policy, handle parent) {
    list l(src.size());
    size_t index = 0;
    for (auto &&value : src) {
      auto value_ = reinterpret_steal<object>(
          value_conv::cast(forward_like<T>(value), policy, parent));
      if (!value_) return handle();
      PyList_SET_ITEM(l.ptr(), (ssize_t)index++,
                      value_.release().ptr());  // steals a reference
    }
    return l.release();
  }

  PYBIND11_TYPE_CASTER(problem::PerDataSpace<Type>, _("PerDataSpace"));
};

/*
 * Type caster for boost::multiprecision::uint128_t.
 *
 * Source: https://stackoverflow.com/questions/54738011/pybind11-boostmultiprecisioncpp-int-to-python
 *
 * FIXME: This could be slow since it involves conversion to string.
 */
template<typename cpp_int_backend>
struct type_caster<mp::number<cpp_int_backend>> {
  PYBIND11_TYPE_CASTER(mp::number<cpp_int_backend>, _("number"));

  bool load(handle src, bool) {
    PyObject* tmp = PyNumber_ToBase(src.ptr(), 16);
    if (!tmp) return false;

    std::string s = py::cast<std::string>(tmp);
    value = mp::uint128_t{s};

    Py_DECREF(tmp);

    return !PyErr_Occurred();
  }

  static handle cast(const mp::number<cpp_int_backend>& src,
                     return_value_policy, handle) {
    std::ostringstream oss;
    oss << std::hex << src;
    return PyLong_FromString(oss.str().c_str(), nullptr, 16);
  }
};
}  // namespace detail
}  // namespace pybind11
