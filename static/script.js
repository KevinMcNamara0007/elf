let running = false;

async function handleSubmit(){
    if(running === false){
        running = true;
        let instruction = document.getElementById("instruction").value;
        await callAPI(instruction)
        running = false;
    }
}
let plan = '';
let result = '';
let resultChunks = false;
async function callAPI(prompt) {
    result = ""
    plan = ""
    resultChunks = false
    try {
        const response = await fetch("/Inference/ask_a_pro_stream", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'token': 'fja0w3fj039jwiej092j0j-9ajw-3j-a9j-ea' // Consider keeping this secure
            },
            body: JSON.stringify({ "output_tokens": 12000, "prompt": prompt })
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status} - ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');


        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            let text = ""
            // Decode the chunk and append it to the result
            text += decoder.decode(value, { stream: true });
            const responseElement = document.getElementById("response");
            const planElement = document.getElementById("plan")
            if(text.includes("!Final!")){
                plan = text
                resultChunks = true
                text = ""
            }
            if(resultChunks === true){
                responseElement.innerText += text
                result += text

                // Scroll to the bottom of the response element
                responseElement.scrollTop = responseElement.scrollHeight;
            }else{
                planElement.innerText += text

                // Scroll to the bottom of the response element
                planElement.scrollTop = planElement.scrollHeight;
            }
            // Log each chunk
            console.log('Received chunk:', decoder.decode(value, { stream: true }));
        }

        // Final log after all chunks have been received
        console.log('Full response:', result);

    } catch (error) {
        console.error('API call failed:', error);
    }
}