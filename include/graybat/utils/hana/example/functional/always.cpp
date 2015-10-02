/*
@copyright Louis Dionne 2015
Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)
 */

#include <boost/hana/functional/always.hpp>
namespace hana = boost::hana;


static_assert(hana::always(1)() == 1, "");
static_assert(hana::always('2')(1, 2, 3) == '2', "");

int main() { }
