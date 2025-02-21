# Smart Documentation & Code Explanation

An AI-powered system that automatically generates clear, detailed documentation and code explanations from existing codebases, making technical details more accessible to developers of all skill levels.

## Features

### Interactive Chat Interface
- AI-powered code explanation with adjustable expertise levels (Beginner/Intermediate/Expert)
- Multiple analysis modes including general explanation, optimization, bug detection, security analysis, and best practices
- Support for multiple AI models (GPT-4 and DeepSeek-R1)
- Persistent chat history with markdown rendering

### Visual Tools
- Code visualization tool for generating flowcharts
- Code performance analyzer for comparing optimized vs unoptimized code
- Modern UI with responsive design and animated background

### Customization Options
- Adjustable expertise levels for explanations
- Multiple task modes for different analysis needs
- Model selection between GPT-4 and DeepSeek-R1
- Dynamic prompt templates based on user preferences

## Technical Stack

### Core Dependencies
- `streamlit`: Web application framework
- `langchain`: LLM integration and prompt management
- `openai`: GPT-4 integration
- `huggingface`: DeepSeek-R1 model integration
- `python-dotenv`: Environment management

### Visual Components
- Custom CSS with animated gradient background
- Matisse blue color scheme
- Responsive layout with tabbed interface
- Chat message styling

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install streamlit langchain-openai langchain-huggingface python-dotenv
```

3. Create a `.env` file with your API keys:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

### Chat Interface
1. Select your preferred model (GPT-4 or DeepSeek-R1) from the sidebar
2. Choose expertise level (Beginner/Intermediate/Expert)
3. Select analysis task type
4. Enter your code or question in the chat input
5. View AI-generated explanations in the chat window

### Code Visualization
1. Navigate to the "Code Visualization" tab
2. Enter your code in the text area
3. Click "Generate Flowchart" to create a visual representation

### Performance Analysis
1. Navigate to the "Code Performance Analyzer" tab
2. Enter optimized and unoptimized code versions
3. Click "Run Codes" to compare performance metrics

## Configuration

### Level-Specific Instructions
The system provides three expertise levels:
- **Beginner**: Detailed explanations with basic concept definitions
- **Intermediate**: Balanced between detail and conciseness
- **Expert**: Advanced concepts and system-level analysis

### Task Types
- General Code Explanation
- Optimization Suggestions
- Bug Detection and Fixes
- Security Vulnerability Analysis
- Best Practices Recommendations

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT-4 integration
- HuggingFace for DeepSeek-R1 model
- Streamlit team for the web framework
- LangChain for LLM integration tools
