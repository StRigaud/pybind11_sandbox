
#include "cleKernel.hpp"

namespace cle
{

void Kernel::ManageDimensions(const char* t_tag)
{
    if(this->m_Sources.size() > 1)
    {
        auto it = this->m_Parameters.find(t_tag);
        if (it != this->m_Parameters.end())
        {
            if(this->m_Parameters.at(t_tag)->nDim() == 3)
            {
                this->m_nDimPrefix = "_3d";
            }
            else if(this->m_Parameters.at(t_tag)->nDim() == 2)
            {
                this->m_nDimPrefix = "_2d";
            }
            else
            {
                this->m_nDimPrefix = "_1d";
            }
        }
        else
        {
            std::cerr << "Kernel : Error in managing data dimensions, could not find \""<< t_tag <<"\" tag in Parameters list." << std::endl;
        }
    }
}
    
const std::string Kernel::LoadPreamble() const
{
    std::string preamble = 
        #include "cle_preamble.h"
    ;
    return preamble;
}

const std::string Kernel::LoadSources() const
{  
    std::string kernel_source = "";
    if(!m_Sources.empty())
    {
        kernel_source = this->m_Sources.at((this->m_KernelName + this->m_nDimPrefix).c_str());
    }
    return kernel_source;
}

const std::string Kernel::BufferDefines(std::string& t_tag, std::string& t_otype, std::string& t_dtype, int t_dim) const
{
    std::string pos_type ("int"), pos ("(pos0)"), dim ("1");
    if (t_dim == 2)
    { 
        dim = "2";
        pos_type = "int2";
        pos = "(pos0, pos1)";
    } 
    else if (t_dim == 3)
    { 
        dim = "3";
        pos_type = "int4";
        pos = "(pos0, pos1, pos2, 0)";
    } 
    std::string abbr = TypeAbbr(t_dtype.c_str());  
    std::string common_header = "\n";
    common_header += "\n#define CONVERT_" + t_tag + "_PIXEL_TYPE clij_convert_" + t_dtype + "_sat";
    common_header += "\n#define IMAGE_" + t_tag + "_PIXEL_TYPE " + t_dtype + "";
    common_header += "\n#define POS_" + t_tag + "_TYPE " + pos_type;
    common_header += "\n#define POS_" + t_tag + "_INSTANCE(pos0,pos1,pos2,pos3) (" + pos_type + ")" + pos;
    std::string array_header = "\n";
    array_header += "\n#define IMAGE_" + t_tag + "_TYPE __global " + t_dtype + "*";
    array_header += "\n#define READ_" + t_tag + "_IMAGE(a,b,c) read_buffer" + dim + "d" + abbr + "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    array_header += "\n#define WRITE_" + t_tag + "_IMAGE(a,b,c) write_buffer" + dim + "d" + abbr + "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    return common_header+array_header;
}

const std::string Kernel::ImageDefines(std::string& t_tag, std::string& t_otype, std::string& t_dtype, int t_dim) const
{
    std::string pos_type ("int"), pos ("(pos0)"), dim ("1");
    if (t_dim == 2)
    { 
        dim = "2";
        pos_type = "int2";
        pos = "(pos0, pos1)";
    } 
    else if (t_dim == 3)
    { 
        dim = "3";
        pos_type = "int4";
        pos = "(pos0, pos1, pos2, 0)";
    } 
    std::string img_type_name;
    if(t_tag.find("dst")!=std::string::npos || t_tag.find("destination")!=std::string::npos || 
       t_tag.find("output")!=std::string::npos)
    {
        img_type_name = "__write_only image" + dim + "d_t";
    }
    else
    {
        img_type_name = "__read_only image" + dim + "d_t";
    }
    std::string abbr = TypeAbbr(t_dtype.c_str());  
    std::string common_header = "\n";
    common_header += "\n#define CONVERT_" + t_tag + "_PIXEL_TYPE clij_convert_" + t_dtype + "_sat";
    common_header += "\n#define IMAGE_" + t_tag + "_PIXEL_TYPE " + t_dtype + "";
    common_header += "\n#define POS_" + t_tag + "_TYPE " + pos_type;
    common_header += "\n#define POS_" + t_tag + "_INSTANCE(pos0,pos1,pos2,pos3) (" + pos_type + ")" + pos;
    std::string image_header = "\n";
    image_header += "\n#define IMAGE_" + t_tag + "_TYPE " + img_type_name;
    image_header += "\n#define READ_" + t_tag + "_IMAGE(a,b,c) read_image" + abbr + "(a,b,c)";
    image_header += "\n#define WRITE_" + t_tag + "_IMAGE(a,b,c) write_image" + abbr + "(a,b,c)";
    return common_header+image_header;
}

const std::string Kernel::LoadDefines() const
{
    std::string defines = "";  
    // image size independent from kernel compilation
    defines = defines + "\n#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## _WIDTH";
    defines = defines + "\n#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key ## _HEIGHT";
    defines = defines + "\n#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## _DEPTH";
    defines = defines + "\n";  
    // loop on parameters
    for (auto itr = this->m_Parameters.begin(); itr != this->m_Parameters.end(); ++itr)
    {
        // only non-scalar argument are relevant
        if (!itr->second->IsObjectType("scalar"))  
        {     
            // compute defines adapted to current argument (type, size, shape, etc.)
            std::string object_tag = itr->first;                      // get argument tag (e.g. dst, destination, output)
            std::string object_type = itr->second->GetObjectType();   // buffer or image
            std::string data_type = itr->second->GetDataType();       // float, double, char, int, etc.
            std::string abbr_type = TypeAbbr(data_type.c_str());       // shorten type f->float, i->int, d->double, etc.
            int ndim = itr->second->nDim();
            
            std::string size_header = "\n";
            size_header = size_header + "\n#define IMAGE_SIZE_" + object_tag + "_WIDTH " + std::to_string(itr->second->Shape()[0]);
            size_header = size_header + "\n#define IMAGE_SIZE_" + object_tag + "_HEIGHT " + std::to_string(itr->second->Shape()[1]);
            size_header = size_header + "\n#define IMAGE_SIZE_" + object_tag + "_DEPTH " + std::to_string(itr->second->Shape()[2]);
            if (itr->second->IsObjectType("buffer"))
            {
                defines += this->BufferDefines(object_tag, object_type, data_type, ndim);
            }   
            else
            {
                defines += this->ImageDefines(object_tag, object_type, data_type, ndim);
            }
            defines += size_header + "\n";
        } // end check if not Scalar Object
    } // end of for loop on hashmap
    return defines;
}

const std::string Kernel::GenerateSources() const
{
    std::string defines_src = LoadDefines();
    std::string kernel_src = LoadSources();
    std::string preambule_src = LoadPreamble();
    return defines_src + "\n" + preambule_src + "\n" + kernel_src;
}

std::string Kernel::TypeAbbr(const char* t_str) const
{
    size_t size = strlen(t_str);
    // if (strncmp("double", t_str, size) == 0) return "d";    // Missing definition type in preamble
    if (strncmp("float", t_str, size) == 0)  return "f";
    if (strncmp("char", t_str, size) == 0)   return "c";
    if (strncmp("uchar", t_str, size) == 0)  return "uc";
    // if (strncmp("int", t_str, size) == 0)    return "i";    // Missing definition type in preamble
    // if (strncmp("uint", t_str, size) == 0)   return "ui";   // Missing definition type in preamble
    // if (strncmp("short", t_str, size) == 0)    return "s";  // Missing definition type in preamble
    // if (strncmp("ushort", t_str, size) == 0)   return "us"; // Missing definition type in preamble
    return "?"; 
}

void Kernel::SetArguments()
{
    for(auto it = this->m_Tags.begin(); it != this->m_Tags.end(); it++ )
    {
        size_t index = it - this->m_Tags.begin();
        if(this->m_Parameters.find(it->c_str()) != this->m_Parameters.end())
        {
            std::string tag = it->c_str();
            if (this->m_Parameters.at(tag)->IsObjectType("buffer"))
            {    
                auto object = std::dynamic_pointer_cast<cle::Buffer>(this->m_Parameters.at(tag));
        
                try
                {
                    this->m_Kernel.setArg(index, *object->Data());
                }
                catch(cl::Error& e)
                {
                    std::cerr << "Kernel : Fail to set argument to kernel." << std::endl;
                    std::cerr << "\tException caught! " << e.what() << " error code " << e.err() << std::endl;
                }
                for (size_t i = 0; i < 3; i++)
                {
                    size_t tempDim = size_t(object->Shape()[i]);
                    this->m_GlobalRange[i] = std::max(m_GlobalRange[i], tempDim);
                }
            }
            else if (this->m_Parameters.at(tag)->IsObjectType("image"))
            {    
                auto object = std::dynamic_pointer_cast<cle::Image>(this->m_Parameters.at(tag));
        
                try
                {
                    this->m_Kernel.setArg(index, *object->Data());
                }
                catch(cl::Error& e)
                {
                    std::cerr << "Kernel : Fail to set argument to kernel." << std::endl;
                    std::cerr << "\tException caught! " << e.what() << " error code " << e.err() << std::endl;
                }
                for (size_t i = 0; i < 3; i++)
                {
                    size_t tempDim = size_t(object->Shape()[i]);
                    this->m_GlobalRange[i] = std::max(m_GlobalRange[i], tempDim);
                }
            }
            else if (this->m_Parameters.at(tag)->IsObjectType("scalar"))  // only float and int used in kernel for now
            {    
                if (this->m_Parameters.at(tag)->IsDataType("float")) {
                    auto object = std::dynamic_pointer_cast< cle::Scalar<float> >(this->m_Parameters.at(tag));
            
                    this->m_Kernel.setArg(index, object->Data()); 
                }
                else if (this->m_Parameters.at(tag)->IsDataType("int")) {
                    auto object = std::dynamic_pointer_cast< cle::Scalar<int> >(this->m_Parameters.at(tag));
                    this->m_Kernel.setArg(index, object->Data()); 
                }
                // else if (this->m_Parameters.at(tag)->IsDataType("double")) {
                //     auto object = std::dynamic_pointer_cast< cle::Scalar<double> >(this->m_Parameters.at(tag));
                //     this->m_Kernel.setArg(index, object->Data()); 
                // }
                // else if (this->m_Parameters.at(tag)->IsDataType("char")) {
                //     autoobject = std::dynamic_pointer_cast< cle::Scalar<char> >(this->m_Parameters.at(tag));
                //     this->m_Kernel.setArg(index, object->Data()); 
                // }
                // else if (this->m_Parameters.at(tag)->IsDataType("short")) {
                //     auto object = std::dynamic_pointer_cast< cle::Scalar<short> >(this->m_Parameters.at(tag));
                //     this->m_Kernel.setArg(index, object->Data()); 
                // }
                // else if (this->m_Parameters.at(tag)->IsDataType("unsigned int")) {
                //     auto object = std::dynamic_pointer_cast< cle::Scalar<unsigned int> >(this->m_Parameters.at(tag));
                //     this->m_Kernel.setArg(index, object->Data()); 
                // }
                // else if (this->m_Parameters.at(tag)->IsDataType("unsigned char")) {
                //     auto object = std::dynamic_pointer_cast< cle::Scalar<unsigned char> >(this->m_Parameters.at(tag));
                //     this->m_Kernel.setArg(index, object->Data());
                // }
                // else if (this->m_Parameters.at(tag)->IsDataType("unsigned short")) {
                //     auto object = std::dynamic_pointer_cast< cle::Scalar<unsigned short> >(this->m_Parameters.at(tag));
                //     this->m_Kernel.setArg(index, object->Data()); 
                // }
            }
        }
    }
}

void Kernel::AddObject(Object& t_ocl_object, const char* t_tag)
{  
    if(std::find(this->m_Tags.begin(), this->m_Tags.end(), t_tag) != this->m_Tags.end())
    {
        auto it = this->m_Parameters.find(t_tag); 
        if (it != this->m_Parameters.end())
        {
            if(t_ocl_object.IsObjectType("image"))
            {
                auto cast_object = dynamic_cast<cle::Image&>(t_ocl_object);
                it->second = std::make_shared<cle::Image>(cast_object);
        
            }
            else
            {
                auto cast_object = dynamic_cast<cle::Buffer&>(t_ocl_object);
                it->second = std::make_shared<cle::Buffer>(cast_object);
        
            }
        }
        else    
        {
            if(t_ocl_object.IsObjectType("image"))
            {
                auto cast_object = dynamic_cast<cle::Image&>(t_ocl_object);
                this->m_Parameters.insert(std::make_pair(t_tag, std::make_shared<cle::Image>(cast_object)));
        
            }
            else
            {
                auto cast_object = dynamic_cast<cle::Buffer&>(t_ocl_object);
                this->m_Parameters.insert(std::make_pair(t_tag, std::make_shared<cle::Buffer>(cast_object)));
        
            }
        }
    }
    else
    {
        std::cerr << "Kernel : Invalid tag '" << t_tag << "' used to add buffer." << std::endl;
    }
}

void Kernel::AddObject(int t_scalar, const char* t_tag)
{
    if(std::find(this->m_Tags.begin(), this->m_Tags.end(), t_tag) != this->m_Tags.end())
    {
        cle::Scalar<int> ocl_scalar(t_scalar);
        auto it = this->m_Parameters.find(t_tag); 
        if (it != this->m_Parameters.end())
        {
            it->second = std::make_shared< cle::Scalar<int> >(ocl_scalar);
        }
        else    
        {
            this->m_Parameters.insert(std::make_pair(t_tag, std::make_shared< cle::Scalar<int> >(ocl_scalar)));
        }
    }
    else
    {
        std::cerr << "Kernel : Invalid tag '" << t_tag << "' used to add int scalar." << std::endl;
    }
}

void Kernel::AddObject(float t_scalar, const char* t_tag)
{
    if(std::find(this->m_Tags.begin(), this->m_Tags.end(), t_tag) != this->m_Tags.end())
    {
        cle::Scalar<float> ocl_scalar(t_scalar);
        auto it = this->m_Parameters.find(t_tag); 
        if (it != this->m_Parameters.end())
        {
            it->second = std::make_shared< cle::Scalar<float> >(ocl_scalar);
        }
        else    
        {
            this->m_Parameters.insert(std::make_pair(t_tag, std::make_shared< Scalar<float> >(ocl_scalar)));
        }
    }
    else
    {
        std::cerr << "Kernel : Invalid tag '" << t_tag << "' used to add float scalar." << std::endl;
    }
}

void Kernel::BuildProgramKernel()
{
    std::string sources = GenerateSources();
    std::hash<std::string> hasher;
    size_t source_hash = hasher(sources);
    if(this->m_CurrentHash != source_hash)  
    {
        if(this->m_gpu->FindProgram(source_hash))
        {
            this->m_Program = this->m_gpu->GetProgram(source_hash);
            this->m_CurrentHash = source_hash;
        }
        else
        {
            try
            {
                this->m_Program = cl::Program(this->m_gpu->Context(), sources);
            }
            catch(cl::Error& e)
            {
                std::cerr << "Kernel : Fail to create program." << std::endl;
                std::cerr << "\tException caught! " << e.what() << " error code " << e.err() << std::endl;
                std::cerr << sources << std::endl;
            }
            try
            {
                this->m_Program.build({this->m_gpu->Device()});
            }
            catch(cl::Error& e)
            {
                std::cerr << "Kernel : Fail to build program \"" << this->m_KernelName << "\" from source." << std::endl;
                std::cerr << "\tException caught! " << e.what() << " error code " << e.err() << std::endl;
                std::cerr << "build log:" << std::endl;
                std::cerr << this->m_Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->m_gpu->Device()) << std::endl;
                std::cerr << sources << std::endl;
            }
            this->m_CurrentHash = source_hash;
            this->m_gpu->AddProgram(this->m_Program, m_CurrentHash);
        }
        std::string fullName = this->m_KernelName + this->m_nDimPrefix;
        try
        {
            this->m_Kernel = cl::Kernel(this->m_Program, fullName.c_str());
        }
        catch(cl::Error& e)
        {
            std::cerr << "Kernel : Fail to create kernel. " << fullName.c_str() << std::endl;
            std::cerr << "\tException caught! " << e.what() << " error code " << e.err() << std::endl;
        }
    }
}

void Kernel::EnqueueKernel()
{
    cl::NDRange globalND(this->m_GlobalRange[0], this->m_GlobalRange[1], this->m_GlobalRange[2]); 
    try
    {
        this->m_gpu->CommandQueue().enqueueNDRangeKernel(this->m_Kernel, cl::NullRange, globalND, cl::NullRange);
    }
    catch(const cl::Error& e)
    {
        std::cerr << "Kernel : Fail to enqueue kernel." << std::endl;
        std::cerr << "\tException caught! " << e.what() << " error code " << e.err() << '\n';
    }
    this->m_gpu->Finish();
}

Kernel::Kernel(std::shared_ptr<GPU> gpu, const char* t_kernel_name, const std::vector<std::string>& t_tags) : 
    m_gpu(gpu), m_KernelName(t_kernel_name), m_Tags(t_tags)
{}

Kernel::~Kernel()
{}

} // namespace clic
