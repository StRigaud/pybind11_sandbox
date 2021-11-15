
#include "clesperanto.hpp"
#include "cleKernelList.hpp"

#include <iostream>
namespace cle
{

Clesperanto::Clesperanto() : m_gpu(std::make_shared<cle::GPU>())
{}

std::shared_ptr<GPU> Clesperanto::Ressources()
{
    return this->m_gpu;
}

void Clesperanto::AddImageAndScalar(Object& t_src, Object& t_dst, float t_scalar)
{
    AddImageAndScalarKernel kernel(this->m_gpu);
    kernel.SetInput(t_src);
    kernel.SetOutput(t_dst);
    kernel.SetScalar(t_scalar);
    kernel.Execute();
}


}