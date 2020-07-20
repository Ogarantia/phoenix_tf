#pragma once
#include <cstdint>

namespace upstride {

/**
 * @brief Represents shapes of a tensor
 * 
 */
class Shape {
    uint8_t size;
    int* shape;

   public:
    /**
    * @brief Construct a new Shape object
    * 
    * @param s Size of the shape array
    * @param _shape Array of shapes
    */
    Shape(int s, const int* _shape) : size(s) {
        shape = new int[s];
        for (int i = 0; i < s; i++) {
            shape[i] = _shape[i];
        }
    }
    /**
     * @brief Construct a new Shape object that creates a s size shape with all dimension to 0
     * 
     * @param s 
     */
    Shape(int s) : size(s) {
        shape = new int[s];
        for (int i = 0; i < s; i++) {
            shape[i] = 0;
        }
    }

    /**
     * @brief Destroy the Shape object
     * 
     */
    ~Shape() { delete[] shape; }
    uint8_t getSize() const { return size; }
    const int* getShapePtr() const { return shape; }
    int operator[](int i) const { return shape[i]; }
    int& operator[](int i) { return shape[i]; }
};

/**
 * @brief Tensor representation using its shape a tensor array
 * 
 * @tparam T Type of the tensor
 */
template <typename T>
class Tensor {
    const Shape shape;
    T* tensor;

   public:
    /**
    * @brief Construct a new Tensor object
    * 
    * @param sh Shape of the tensor
    * @param t Tensor
    */
    Tensor(const Shape& sh, T* t) : shape(sh.getSize(), sh.getShapePtr()),
                                    tensor(t) {}
    /**
     * @brief Get the pointer to the Tensor object 
     * 
     * @return T* Pointer to tensor
     */
    T* getTensorPtr() { return tensor; }

    /**
     * @brief Get the Shape object
     * 
     * @return const Shape& 
     */
    const Shape& getShape() const { return shape; }
};

}  // namespace upstride