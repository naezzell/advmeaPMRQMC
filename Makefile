CXX ?= g++
MPICXX ?= mpicxx
CXXFLAGS ?= -O3 -std=c++11

.PHONY: all clean test validate-pt

all: prepare.bin PMRQMC.bin PMRQMC_mpi.bin PMRQMC_pt_mpi.bin

prepare.bin: prepare.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

PMRQMC.bin: PMRQMC.cpp mainqmc.hpp divdiff.hpp hamiltonian.hpp parameters.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

PMRQMC_mpi.bin: PMRQMC_mpi.cpp mainqmc.hpp divdiff.hpp hamiltonian.hpp parameters.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

PMRQMC_pt_mpi.bin: PMRQMC_pt_mpi.cpp pt_schedule.hpp mainqmc.hpp divdiff.hpp hamiltonian.hpp parameters.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

test: tests/pt_components_test
	./tests/pt_components_test /tmp/pmrqmc_pt_schedule_test.txt
	python3 tests/pt_python_test.py

validate-pt:
	python3 experiments/validate_pt_minimal.py --oversubscribe

tests/pt_components_test: tests/pt_components_test.cpp pt_schedule.hpp divdiff.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f prepare.bin PMRQMC.bin PMRQMC_mpi.bin PMRQMC_pt_mpi.bin tests/pt_components_test
