#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "../pmedian/instance.h"
#include "../pmedian/solution.h"
#include "../kmedoids/medoids.h"
#include "../kmedoids/kmedoids.h"
#include "../tools/trie.h"
#include "../ts/ts.h"

namespace py = pybind11;

PYBIND11_MODULE (xopt, m) {
    m.doc() = "Python bindings for the XOpt tabu search and k-medoids toolkit.";

    py::class_<Instance> (m, "Instance")
        .def(py::init <> ())

        .def(py::init <const std::string&> () , py::arg("filename"))
        .def("load"          , &Instance::load, py::arg("filename"))

        .def("floyd_warshall", &Instance::floyd_warshall)
        .def("describe"      , &Instance::describe      )

        .def("get_n", &Instance::get_n)
        .def("get_p", &Instance::get_p)
        .def("__getitem__",
            [] (const Instance& instance, py::tuple idx) {
                if (idx.size() != 2)
                    throw py::index_error("Instance indices must be a pair (i, j)");

                int i = idx[0].cast <int> ();
                int j = idx[1].cast <int> ();

                if (i < 0 || i >= instance.get_n() ||
                    j < 0 || j >= instance.get_n())
                    throw py::index_error("Instance index out of range");

                return instance(i, j);
            }
        );

    py::class_<Solution> (m, "Solution")
        .def(py::init <> ())

        .def_readwrite("cost"      , &Solution::cost      )
        .def_readwrite("facilities", &Solution::facilities)
        .def("describe", &Solution::describe);

    py::class_<Medoids> (m, "Medoids")
        .def(py::init <> ())

        .def_readwrite("cost"   , &Medoids::cost   )
        .def_readwrite("medoids", &Medoids::medoids);

    py::class_<SolutionTrie, std::shared_ptr <SolutionTrie>> (m, "SolutionTrie")
        .def(py::init <int, int> (), py::arg("n"), py::arg("p"))

        .def("update"       , &SolutionTrie::update       , py::arg("solution"))
        .def("update_mask"  , &SolutionTrie::update_mask  , py::arg("mask"    ))
        .def("contains"     , &SolutionTrie::contains     , py::arg("solution"))
        .def("contains_mask", &SolutionTrie::contains_mask, py::arg("mask"    ))
        .def("get_all_solutions",
            [] (const SolutionTrie& trie, const Instance& instance) {
                return trie.get_all_solutions(instance);
            },
            py::arg("instance")
        );

    py::class_<TSResult> (m, "TSResult")
        .def(py::init <std::shared_ptr <SolutionTrie>> (), py::arg("long_term"))

        .def_readwrite("best"     , &TSResult::best     )
        .def_readonly ("long_term", &TSResult::long_term);

    m.def (
        "kmedoids",
        &kmedoids ,
        py::arg("instance"),
        py::arg("max_iter") = 10,
        py::arg("restarts") = 5
    );

    m.def (
        "tspmed",
        &tspmed ,
        py::arg("instance"),
        py::arg("medoids" ),
        py::arg("iter_factor") = 2,
        py::arg("long_term")
    );
}