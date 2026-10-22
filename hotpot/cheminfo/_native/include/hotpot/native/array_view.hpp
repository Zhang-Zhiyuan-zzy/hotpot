#pragma once

#include <array>
#include <cstddef>
#include <vector>


namespace hotpot::native {


template <typename Value>
class ArrayView {
public:
    constexpr ArrayView() noexcept = default;

    constexpr ArrayView(const Value* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    template <std::size_t Size>
    constexpr ArrayView(const std::array<Value, Size>& values) noexcept
        : data_(values.data()), size_(values.size()) {}

    explicit ArrayView(const std::vector<Value>& values) noexcept
        : data_(values.data()), size_(values.size()) {}

    constexpr const Value* data() const noexcept { return data_; }
    constexpr const Value* begin() const noexcept { return data_; }
    constexpr const Value* end() const noexcept {
        return size_ == 0 ? data_ : data_ + size_;
    }
    constexpr const Value& operator[](std::size_t index) const noexcept {
        return data_[index];
    }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }

private:
    const Value* data_ = nullptr;
    std::size_t size_ = 0;
};


}  // namespace hotpot::native
