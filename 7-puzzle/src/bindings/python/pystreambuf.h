/*
Based on https://gist.github.com/asford/544323a5da7dddad2c9174490eb5ed06
but was heavily modified.

Original license text
---------------------

This component utilizes components derived from cctbx, available at
http://cci.lbl.gov/cctbx_sources/boost_adaptbx/python_streambuf.h

*** License agreement ***

cctbx Copyright (c) 2006, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to
receipt of any required approvals from the U.S. Dept. of Energy).  All
rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes,
patches, or upgrades to the features, functionality or performance of
the source code ("Enhancements") to anyone; however, if you choose to
make your Enhancements available either publicly, or directly to
Lawrence Berkeley National Laboratory, without imposing a separate
written license agreement for such Enhancements, then you hereby grant
the following license: a  non-exclusive, royalty-free perpetual license
to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.

*/

#pragma once

#include <pybind11/pybind11.h>

#include <iostream>
#include <sstream>

namespace py = pybind11;


namespace pybind11::detail {
    template <> struct type_caster<std::istream> {

        bool load(handle src, bool) {

            auto read_method = getattr(src, "read", py::none());

            if (read_method.is_none()){
            return false;
            }

            value = std::make_unique<std::istringstream>(py::reinterpret_borrow<py::bytes>(read_method()));

            return true;
        }

        static handle cast(std::istream &src, return_value_policy policy, handle parent) {
            return none().release();
        }

        static constexpr auto name = const_name("io.BytesIO");
        operator std::istream*() { return value.get(); }
        operator std::istream&() { return *value; }
        template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

        std::unique_ptr<std::istringstream> value;

    };

    template <> struct type_caster<std::ostream> {

        struct pyostringstream : std::ostringstream {
            py::object write_method;
            bool is_binary;

            pyostringstream(py::object write_method_, bool binary) 
                : std::ostringstream{}, write_method{write_method_}, is_binary{binary} {};

            ~pyostringstream() {
                // Get the string buffer from the C++ side
                std::string s = this->str();
                if (s.empty()) return;

                try {
                    if (is_binary) {
                        this->write_method(py::bytes(s));
                    } else {
                        // Convert C++ string (assumed UTF-8) to Python str
                        this->write_method(py::str(s));
                    }
                } catch (py::error_already_set &e) {
                    // Destructors shouldn't throw; print or discard error
                    py::print("Error writing to Python stream: ", e.what());
                }
            }
        };

        bool load(handle src, bool) {
            auto write_method = getattr(src, "write", py::none()); 
            if (write_method.is_none()) return false;

            // Check if the stream is binary or text. 
            // We look for 'b' in the mode or check for TextIOBase.
            bool binary = true;
            if (py::hasattr(src, "encoding") || py::isinstance(src, py::module_::import("io").attr("TextIOBase"))) {
                binary = false;
            }

            value = std::make_unique<pyostringstream>(write_method, binary);
            return true;
        }

        static handle cast(std::ostream &, return_value_policy, handle) {
            return none().release();
        }

        // Change name to reflect it's more general now
        static constexpr auto name = const_name("typing.IO");
        
        operator std::ostream*() { return value.get(); }
        operator std::ostream&() { return *value; }
        
        template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
        std::unique_ptr<pyostringstream> value;
    };
}