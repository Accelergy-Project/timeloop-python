#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() {}

  bool Push(const T& val) {
    std::lock_guard<std::mutex> enq_lock(m_);
    q_.push(val);
    cv_.notify_all();
    return true;
  }

  bool Pop(T& val) {
    std::unique_lock<std::mutex> deq_lock(m_);
    if (q_.empty()) {
      return false;
    }
    val = std::move(q_.front());
    q_.pop();
    return true;
  }

  template <typename Predicate>
  bool Pop(T& val, Predicate stop_waiting) {
    std::unique_lock<std::mutex> deq_lock(m_);
    while (q_.empty() && !stop_waiting()) {
      cv_.wait(deq_lock);
    }
    if (q_.empty()) {
      return false;
    }
    val = std::move(q_.front());
    q_.pop();
    return true;
  }

  void NotifyAll() { cv_.notify_all(); }

 private:
  std::queue<T> q_;
  std::mutex m_;
  std::condition_variable cv_;
};
