// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sellp.hpp>

#include "core/test/utils.hpp"


namespace {


class SellpSpmm : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = int;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Mtx = gko::matrix::Sellp<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    static constexpr gko::size_type num_rows = 200;
    static constexpr gko::size_type num_cols = 150;
    static constexpr gko::size_type num_rhs = 16;
    static constexpr value_type tolerance = 1e-12;

    SellpSpmm() : rand_engine(42) {}

    void SetUp() override
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown() override
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(gko::size_type rows, gko::size_type cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            rows, cols,
            std::uniform_int_distribution<>(min_nnz_row,
                                            static_cast<int>(cols)),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data()
    {
        rand_engine.seed(42);
        auto csr = gen_mtx<Csr>(num_rows, num_cols, 5);
        mtx = Mtx::create(ref);
        csr->convert_to(mtx);
        y = gen_mtx<Vec>(num_cols, num_rhs, 1);
        expected = Vec::create(ref, gko::dim<2>{num_rows, num_rhs});
        dmtx = Mtx::create(omp);
        csr->convert_to(dmtx);
        dy = Vec::create(omp);
        dy->copy_from(y);
        dresult = Vec::create(omp, gko::dim<2>{num_rows, num_rhs});
    }

    void assert_variant_matches_ref(int variant)
    {
        mtx->apply(y, expected);
        dmtx->set_spmm_version(variant);
        dmtx->apply(dy, dresult);
        GKO_ASSERT_MTX_NEAR(dresult, expected, tolerance);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dresult;
};


TEST_F(SellpSpmm, SpmmV1IsEquivalentToRef)
{
    set_up_apply_data();
    assert_variant_matches_ref(1);
}


TEST_F(SellpSpmm, SpmmV2IsEquivalentToRef)
{
    set_up_apply_data();
    assert_variant_matches_ref(2);
}


}  // namespace
