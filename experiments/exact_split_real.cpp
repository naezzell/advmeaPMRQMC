// Dense exact reference for real Pauli split Hamiltonians on small instances.
// macOS Accelerate supplies the symmetric eigensolver, avoiding a Python
// numerical-package dependency in reproducible validation campaigns.
#include <Accelerate/Accelerate.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Term { double coefficient; std::vector<std::pair<int,char> > operators; };

static std::vector<Term> read_terms(const std::string& filename){
    std::ifstream input(filename.c_str());
    if(!input) throw std::runtime_error("cannot open term file: " + filename);
    std::vector<Term> terms; std::string line;
    while(std::getline(input,line)){
        std::string::size_type comment = line.find('#');
        if(comment != std::string::npos) line.erase(comment);
        std::istringstream row(line); Term term;
        if(!(row >> term.coefficient)) continue;
        int site; char pauli;
        while(row >> site >> pauli) term.operators.push_back(std::make_pair(site-1,pauli));
        if(!row.eof()) throw std::runtime_error("invalid Pauli row: " + line);
        terms.push_back(term);
    }
    return terms;
}

static void add_term(std::vector<double>& matrix, int n, const Term& term, double scale){
    const int dimension = 1 << n;
    for(int state=0; state<dimension; state++){
        int target = state; double phase = 1.0;
        for(size_t k=0;k<term.operators.size();k++){
            int site = term.operators[k].first; char pauli = term.operators[k].second;
            bool bit = (state >> site) & 1;
            if(pauli == 'X') target ^= 1 << site;
            else if(pauli == 'Z') phase *= bit ? -1.0 : 1.0;
            else throw std::runtime_error("exact_split_real supports only X and Z");
        }
        matrix[static_cast<size_t>(target)*dimension + state] += scale * term.coefficient * phase;
    }
}

int main(int argc, char** argv){
    if(argc != 7){
        std::cerr << "usage: exact_split_real N H_fixed H_gamma observable beta gamma\n";
        return 2;
    }
    try{
        const int n = std::atoi(argv[1]);
        const std::vector<Term> fixed = read_terms(argv[2]);
        const std::vector<Term> gamma_terms = read_terms(argv[3]);
        const std::vector<Term> observable = read_terms(argv[4]);
        const double beta = std::atof(argv[5]), gamma = std::atof(argv[6]);
        const int dimension = 1 << n;
        std::vector<double> h(static_cast<size_t>(dimension)*dimension,0.0), o(h.size(),0.0);
        for(size_t i=0;i<fixed.size();i++) add_term(h,n,fixed[i],1.0);
        for(size_t i=0;i<gamma_terms.size();i++) add_term(h,n,gamma_terms[i],gamma);
        for(size_t i=0;i<observable.size();i++) add_term(o,n,observable[i],1.0);
        char jobz = 'V', uplo = 'U'; int nn=dimension, lda=dimension, info=0, lwork=-1;
        double query=0.0; std::vector<double> eigenvalues(dimension);
        dsyev_(&jobz,&uplo,&nn,h.data(),&lda,eigenvalues.data(),&query,&lwork,&info);
        if(info != 0) throw std::runtime_error("dsyev workspace query failed");
        lwork = static_cast<int>(query)+8; std::vector<double> work(lwork);
        dsyev_(&jobz,&uplo,&nn,h.data(),&lda,eigenvalues.data(),work.data(),&lwork,&info);
        if(info != 0) throw std::runtime_error("dsyev failed with info=" + std::to_string(info));
        double minimum = eigenvalues[0], partition=0.0, energy=0.0, magnetization=0.0;
        for(int column=0;column<dimension;column++){
            double weight=std::exp(-beta*(eigenvalues[column]-minimum));
            double observable_value=0.0;
            // h now stores eigenvectors column-major.  Rebuild O in the same
            // basis and evaluate <v|O|v> without assuming O is diagonal.
            observable_value=0.0;
            for(int row=0;row<dimension;row++) for(int col=0;col<dimension;col++)
                observable_value += h[static_cast<size_t>(column)*dimension+row] * o[static_cast<size_t>(col)*dimension+row] * h[static_cast<size_t>(column)*dimension+col];
            partition += weight; energy += weight*eigenvalues[column]; magnetization += weight*observable_value;
        }
        std::cout << std::setprecision(17) << beta << ' ' << gamma << ' '
                  << energy/partition << ' ' << magnetization/partition << '\n';
    }catch(const std::exception& error){ std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
