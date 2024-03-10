#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

// Declare executeModelCall function before it's used
void executeModelCall(const std::string& command);

class ExpertFramework {
public:
    ExpertFramework() {}

    std::thread handleRequest(const std::string& prompt) {
        static std::atomic<unsigned long long> requestCounter(0);
        unsigned long long requestId = ++requestCounter;
        return std::thread([this, prompt, requestId] { this->processRequest(prompt, requestId); });
    }

private:
    static std::mutex consoleMutex;

    std::string toLower(const std::string& s) {
        std::string result;
        std::transform(s.begin(), s.end(), std::back_inserter(result),
                       [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    std::string identifyRequestType(const std::string& prompt) {
        std::string promptLower = toLower(prompt);
        std::vector<std::string> developerKeywords = {"java", "python", "c#", "c++", "function"};
        for (const auto& keyword : developerKeywords) {
            if (promptLower.find(keyword) != std::string::npos) {
                return "developer";
            }
        }

        std::vector<std::string> mathKeywords = {"calculate", "evaluate", "differentiate", "integrate", "predict"};
        for (const auto& keyword : mathKeywords) {
            if (promptLower.find(keyword) != std::string::npos) {
                return "math";
            }
        }

        std::vector<std::string> inferenceKeywords = {"predict", "hypothesize", "forecast"};
        for (const auto& keyword : inferenceKeywords) {
            if (promptLower.find(keyword) != std::string::npos) {
                return "inference";
            }
        }

        return "general";
    }

    int calculateContextTokenSize(const std::string& prompt) {
        int averageWordLength = 6;
        return prompt.size() / averageWordLength;
    }

    std::string buildModelCall(const std::string& requestType, const std::string& prompt, int tokenSize) {
        std::string basePath = "/Users/loki/Desktop/worksurface/ai_lab/models/foundation/";
        std::map<std::string, std::string> modelPaths = {
            {"general", "mistral/mistral-7b-instruct-v0.2.Q2_K.gguf"},
            {"developer", "mistral/mistral-7b-instruct-v0.2.Q2_K.gguf"},
            {"math", "mistral/mistral-7b-instruct-v0.2.Q2_K.gguf"},
            {"inference", "mistral/mistral-7b-instruct-v0.2.Q2_K.gguf"}
        };

        std::string modelPath = basePath + modelPaths[requestType];
        std::string command = "./main -m " + modelPath + " -p \"" + prompt + "\" -n 5000 -e -ngl " + std::to_string(tokenSize) + " -t 8";
        return command;
    }

    void processRequest(const std::string& prompt, unsigned long long requestId) {
        std::lock_guard<std::mutex> lock(consoleMutex);
        if (prompt.empty()) {
            std::cerr << "Error: The prompt is empty. [Request ID: " << requestId << "]" << std::endl;
            return;
        }

        std::string requestType = identifyRequestType(prompt);
        int tokenSize = calculateContextTokenSize(prompt);
        std::cout << "Request ID: " << requestId << " - Type: " << requestType << " - Estimated token size for the input request: " << tokenSize << std::endl;

        std::string modelCall = buildModelCall(requestType, prompt, tokenSize);
        // Now actually execute the model call
        executeModelCall(modelCall);
    }
};

std::mutex ExpertFramework::consoleMutex;

void executeModelCall(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    std::cout << result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <prompt>" << std::endl;
        return 1;
    }

    std::string prompt(argv[1]); // Take the first argument as the prompt

    ExpertFramework framework;
    std::thread requestThread = framework.handleRequest(prompt);
    requestThread.join(); // Wait for the request to be processed

    return 0;
}
