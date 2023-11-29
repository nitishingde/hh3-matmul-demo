#ifndef HH3_MATMUL_STATES
#define HH3_MATMUL_STATES

#include "data.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class InputState: public hh::AbstractState<
        1,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>,
        MatrixContainer<MatrixType, IdA>,
        MatrixContainer<MatrixType, IdB>,
        MatrixContainer<MatrixType, IdC>,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>
    > {
private:
    using MatrixA = MatrixContainer<MatrixType, IdA>;
    using MatrixB = MatrixContainer<MatrixType, IdB>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;

public:
    void execute(std::shared_ptr<Triplet> triplet) override {
        auto matrixA = std::get<std::shared_ptr<MatrixA>>(*triplet);
        auto matrixB = std::get<std::shared_ptr<MatrixB>>(*triplet);
        auto matrixC = std::get<std::shared_ptr<MatrixC>>(*triplet);
        assert(matrixA->matrixNumColTiles() == matrixB->matrixNumRowTiles());
        assert(matrixA->matrixNumRowTiles() == matrixC->matrixNumRowTiles());
        assert(matrixB->matrixNumColTiles() == matrixC->matrixNumColTiles());

        this->addResult(matrixA);
        this->addResult(matrixB);
        this->addResult(matrixC);
        this->addResult(triplet);
    };
};

#endif //HH3_MATMUL_STATES
