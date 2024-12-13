#include <iostream>
#include <vector>
#include <string>
#include <eigen/Eigen/Dense>
#include <torch/torch.h>

using namespace std;

int main()
{
    Eigen::Matrix2d m;  
    m(0, 0) = 3;  
    m(1, 0) = 2.5;  
    m(0, 1) = -1;  
    m(1, 1) = m(1, 0) + m(0, 0);  

    std::cout << m << std::endl; 

    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;

    torch::Tensor tensor = torch::rand({2, 3});
    tensor = tensor.cuda();
    cout << tensor << endl;
}

