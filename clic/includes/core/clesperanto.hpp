
#ifndef __clesperanto_hpp
#define __clesperanto_hpp

#include "cleGPU.hpp"
#include "cleBuffer.hpp"
#include "cleImage.hpp"
#include "cleObject.hpp"

#include <type_traits>
#include <iostream>

namespace cle
{

class Clesperanto
{
private:
    std::shared_ptr<cle::GPU> m_gpu;

public:
    Clesperanto();
    ~Clesperanto() = default;

    template<class T =float>
    cle::Buffer Create(const std::array<size_t,3>& ={1,1,1}) const;
    template<class T =float>
    cle::Image CreateImage(const std::array<size_t,3>& ={1,1,1}) const;
    template<class T =float>
    cle::Buffer Push(std::vector<T>& ={0}, const std::array<size_t,3>& ={1,1,1}) const;
    template<class T =float>
    cle::Image PushImage(std::vector<T>& ={0}, const std::array<size_t,3>& ={1,1,1}) const;
    template<class T =float>
    std::vector<T> Pull(cle::Buffer&) const;
    template<class T =float>
    std::vector<T> PullImage(cle::Image&) const;

    std::shared_ptr<GPU> Ressources();

    void AddImageAndScalar(Object&, Object&, float=0);
};


    template<class T>
    cle::Buffer Clesperanto::Create(const std::array<size_t,3>& t_shape) const
    {
        return this->m_gpu->CreateBuffer<T>(t_shape);
    }

    template<class T>
    cle::Image Clesperanto::CreateImage(const std::array<size_t,3>& t_shape) const
    {
        return this->m_gpu->CreateImage<T>(t_shape);
    }

    template<class T>
    cle::Buffer Clesperanto::Push(std::vector<T>& t_array, const std::array<size_t,3>& t_shape) const
    {
        return this->m_gpu->PushBuffer<T>(t_array, t_shape);
    }

    template<class T>
    cle::Image Clesperanto::PushImage(std::vector<T>& t_array, const std::array<size_t,3>& t_shape) const
    {
        return this->m_gpu->PushImage<T>(t_array, t_shape);
    }

    template<class T>
    std::vector<T> Clesperanto::Pull(cle::Buffer& t_data) const
    {
        return this->m_gpu->Pull<T>(t_data);
    }

    template<class T>
    std::vector<T> Clesperanto::PullImage(cle::Image& t_data) const
    {
        return this->m_gpu->Pull<T>(t_data);
    }


} // namespace cle

#endif //__clesperanto_hpp