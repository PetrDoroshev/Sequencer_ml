#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <vector>

int main() {
    torch::jit::script::Module module;

    try {
        module = torch::jit::load("../ar_vae.pt");
        module.eval();
        std::cout << "Модель успешно загружена.\n";
    } catch (const c10::Error& e) {
        std::cerr << "Ошибка загрузки модели: " << e.what() << "\n";
        return -1;
    }

    const int64_t bos_id = 1;
    const int64_t latent_dim = 64;
    const int64_t max_len = 100;
    const double temperature = 0.9;

    torch::NoGradGuard no_grad;

    // generate ожидает start_token формы (batch,) или (batch, 1)
    torch::Tensor start_token =
        torch::tensor({bos_id}, torch::TensorOptions().dtype(torch::kInt64));  // (1,)

    torch::Tensor z =
        torch::randn({1, latent_dim}, torch::TensorOptions().dtype(torch::kFloat32));  // (1, latent_dim)

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(start_token);
    inputs.emplace_back(z);          // или torch::jit::IValue() чтобы z сэмплился внутри generate
    inputs.emplace_back(max_len);
    inputs.emplace_back(temperature);

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor generated = module.get_method("generate")(inputs).toTensor();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Время генерации: " << duration.count() << " мс\n";

        std::cout << "generated.sizes(): " << generated.sizes() << "\n";
        std::cout << generated << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "Ошибка при выполнении generate: " << e.what() << "\n";
        return -1;
    }

    return 0;
}

