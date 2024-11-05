### **Inference Endpoints**

#### **`POST /Inference/ask_an_expert`**

- **Description**: Ask a question to an LLM. Requires either prompt or messages.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `AskExpertRequest`:
    ```python
    class AskExpertRequest(BaseModel):
        temperature: float = 0.05  # Optional, LLM temperature
        rules: str = "You are a virtual assistant."  # Optional, role the LLM should play
        top_k: int = 40  # Optional, number of words to consider for next token
        top_p: float = .95  # Optional, percentage to limit next token generation to
        messages: Optional[List[Message]] = None  # Optional, message history if prompt is not included
        prompt: Optional[str] = None  # Optional, prompt to ask if messages are not included
    ```
- **Response**: The response from the LLM.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Provide messages or prompt
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /Inference/ask_an_expert_stream`**

- **Description**: Ask a question to an LLM and stream the response.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `AskExpertRequest`:
    ```python
    class AskExpertRequest(BaseModel):
        temperature: float = 0.05  # Optional, LLM temperature
        rules: str = "You are a virtual assistant."  # Optional, role the LLM should play
        top_k: int = 40  # Optional, number of words to consider for next token
        top_p: float = .95  # Optional, percentage to limit next token generation to
        messages: Optional[List[Message]] = None  # Optional, message history if prompt is not included
        prompt: Optional[str] = None  # Optional, prompt to ask if messages are not included
    ```
- **Response**: chunked str
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Provide messages or prompt
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /Inference/classify`**

- **Description**: Classify your prompt.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `ClassifyRequest`:
    ```python
    class ClassifyRequest(BaseModel):
        prompt: str  # Required, prompt to classify
    ```
- **Response**: str
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---