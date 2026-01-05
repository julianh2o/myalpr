// Ollama client configuration
client<llm> OllamaVision {
  provider "openai-generic"
  options {
    base_url env.OLLAMA_URL
    model env.OLLAMA_VISION_MODEL
  }
}

// Function to read license plate from image
function ReadPlate(
  frame: image,
  known_plates: string[] | null
) -> string {
  client OllamaVision
  prompt #"
    {{ _.role("user") }}

    This is a license plate image. Please read the license plate number/text.

    {% if known_plates %}
    The plate is potentially one of these known plates: {{ known_plates | join(', ') }}.
    Use this as a guide if the image is unclear or to help with similar looking characters.
    {% endif %}

    IMPORTANT: Return ONLY the alphanumeric characters in the plate or N/A if the plate is absent or cannot be read.
    IMPORTANT: Do not include spaces, punctuation, preamble, explanation, or description

    {{ frame }}
  "#
}

// Test example (optional)
test TestReadPlate {
  functions [ReadPlate]
  args {
    frame {
      url "https://example.com/sample-plate.jpg"
    }
    known_plates ["ABC123", "XYZ789"]
  }
}
