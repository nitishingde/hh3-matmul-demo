#ifndef HEDGEHOG_TUTORIALS_SERIALIZATION_H
#define HEDGEHOG_TUTORIALS_SERIALIZATION_H


#include <string>

class Serialization {
private:
public:
    Serialization() = default;
    virtual ~Serialization() = default;
    [[nodiscard]] virtual std::string serialize() const = 0;
    virtual void deserialize(std::istream&&) = 0;
};


#endif //HEDGEHOG_TUTORIALS_SERIALIZATION_H
