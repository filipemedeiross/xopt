#include <pybind11/pybind11.h>
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

        .def("get_n"   , &Instance::get_n)
        .def("get_p"   , &Instance::get_p)
        .def("__call__", &Instance::operator(), py::arg("i"), py::arg("j"));

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

        .def("contains"           , &SolutionTrie::contains           , py::arg("solution"))
        .def("contains_and_update", &SolutionTrie::contains_and_update, py::arg("solution"))
        .def("get_all_solutions"  , &SolutionTrie::get_all_solutions)

        .def_static(
            "get_global_instance",
            &SolutionTrie::get_global_instance,
            py::arg("n"),
            py::arg("p")
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
        py::arg("iter_factor") = 2
    );
}