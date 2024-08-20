### **CRUD Endpoints**

#### **`POST /CRUD/get_record`**

- **Description**: Fetches records based on title and text contained in the document.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `GetRecordRequest`:
    ```python
    class GetRecordRequest(BaseModel):
        collection_name: str  # Required
        titles: Optional[str] = None  # Optional, titles separated by :::
        text_to_find: Optional[str] = None  # Optional, text to look for in documents
        metadata: Optional[list] = None  # Optional, not yet implemented
        limit: Optional[int] = None  # Optional, number of records to return
    ```
- **Response**: Returns the matched records from the specified collection.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /CRUD/add_record`**

- **Description**: Adds new records based on title and text contained in the document.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `AddRecordRequest`:
    ```python
    class AddRecordRequest(BaseModel):
        titles: str  # Required, titles separated by :::
        contents: str  # Required, contents separated by :::
        collection_name: str  # Required, singular collection name
        metadata: Optional[list] = None  # Optional, not yet implemented
    ```
- **Response**: Adds the records to the specified collection.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /CRUD/update_record`**

- **Description**: Updates records based on title and text contained in the document.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `UpdateRecordRequest`:
    ```python
    class UpdateRecordRequest(BaseModel):
        titles: str  # Required, titles separated by :::
        contents: str  # Required, contents separated by :::
        collection_name: str  # Required, singular collection name
    ```
- **Response**: Updates the specified records in the collection.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /CRUD/delete_record`**

- **Description**: Removes records based on title and collection name.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `DeleteRecordRequest`:
    ```python
    class DeleteRecordRequest(BaseModel):
        titles: str  # Required, titles separated by :::
        collection_name: str  # Required, singular collection name
    ```
- **Response**: Deletes the specified records from the collection.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /CRUD/get_collections`**

- **Description**: Returns all available collections.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Response**: A list of all available collections.
- **Status Codes**:
  - `200 OK`: Success
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /CRUD/add_collection`**

- **Description**: Adds a new collection.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `AddCollectionRequest`:
    ```python
    class AddCollectionRequest(BaseModel):
        collection_name: str  # Required, singular collection name
    ```
- **Response**: Adds the specified collection.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /CRUD/delete_collection`**

- **Description**: Removes a specified collection.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `AddCollectionRequest`:
    ```python
    class AddCollectionRequest(BaseModel):
        collection_name: str  # Required, singular collection name
    ```
- **Response**: Deletes the specified collection.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

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
- **Response**: Streams the response from the LLM as plain text.
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
- **Response**: The classification result.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---

#### **`POST /Inference/semantic_search`**

- **Description**: Perform a semantic search.
- **Headers**:
  - `token` (required): The authorization token. Default value is set to `NO_TOKEN`.
- **Request Body**:
  - `SemanticSearchRequest`:
    ```python
    class SemanticSearchRequest(BaseModel):
        query: str  # Required, query to search against
        collection_name: str  # Required, collection to retrieve records from
        max_results: int = 5  # Optional, max number of results to return
    ```
- **Response**: The matched records from the specified collection.
- **Status Codes**:
  - `200 OK`: Success
  - `400 Bad Request`: Bad input
  - `403 Forbidden`: Invalid token
  - `500 Internal Server Error`: Internal server error

---