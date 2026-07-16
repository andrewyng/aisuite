# LM Studio

LM Studio allows users to locally host open-source models available in [their model catalog](https://lmstudio.ai/models). 
It also provides a web portal with a ChatGPT-like interface.
Once an LM Studio instance is locally running in your setup (default `http://localhost:1234`), you can use the `aisuite` API for chat completions as shown below.
No API Key is needed for these locally hosted models.

## Create a Chat Completion

Sample code:
```python
import aisuite as ai

def main():
    # Set the API URL to remote NGA2 server
    client = ai.Client(
        provider_configs={
            "lmstudio": {
                "api_url": "http://localhost:1234",
                "timeout": 300,
            }
        }
    )
    messages = [
        {
            "role": "system", 
            "content": "Be verbose"
        },
        {
            "role": "user", 
            "content": "Tell me something about University of Michigan's CSE department."
        },
    ]

    lmstudio_llama = "lmstudio:llama-3.2-3b-instruct"

    response = client.chat.completions.create(
        model=lmstudio_llama, 
        messages=messages, 
        temperature=0.75,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
```

You can expect a response like the following:

```markdown
The Computer Science and Engineering (CSE) Department at the University of Michigan is one of the most prestigious and highly-regarded computer science programs in the world. Located in the heart of Ann Arbor, Michigan, the Department of CSE is a leading institution for undergraduate and graduate education in the field of computer science.

With a rich history dating back to 1940, the CSE Department at the University of Michigan has a long tradition of academic excellence, cutting-edge research, and innovative teaching. The department is composed of over 70 faculty members, many of whom are prominent researchers in their fields, and has a student body of around 500 undergraduate majors and 1,000 graduate students.

The CSE Department offers a wide range of undergraduate and graduate degree programs, including Bachelor of Science in Computer Science, Bachelor of Arts in Computer Science, Master of Science in Computer Science, Master of Engineering in Computer Science, and Ph.D. in Computer Science. These programs are designed to provide students with a comprehensive education in computer science, including a strong foundation in mathematics, computer systems, algorithms, computer networks, and software engineering.

The department is particularly renowned for its research programs in areas such as artificial intelligence, computer vision, natural language processing, robotics, and data science. The CSE Department has a strong research focus, and its faculty members are actively engaged in research projects, partnerships, and collaborations with industry, government, and academia.

One of the unique aspects of the CSE Department at the University of Michigan is its strong commitment to interdisciplinary research and education. The department has established partnerships with various academic departments across the university, including physics, mathematics, and engineering, to provide students with a well-rounded education that incorporates multiple disciplines.

The CSE Department also has a strong focus on industry collaboration and engagement. The department has established the University of Michigan's College of Engineering, which provides students with opportunities to engage in research, internships, and co-op programs with top industry partners.

Overall, the Computer Science and Engineering Department at the University of Michigan is a world-class institution that provides students with a world-class education, innovative research opportunities, and strong industry connections. Its highly-regarded faculty, cutting-edge research programs, and strong industry partnerships make it an attractive destination for students interested in pursuing a career in computer science.

Some of the key statistics and achievements of the CSE Department at the University of Michigan include:

* Ranked #5 in the US News & World Report's Best Undergraduate Computer Science Programs (2022)
* Ranked #10 in the QS World University Rankings by Subject: Computer Science (2022)
* 97% of undergraduate graduates find employment or continue their education within six months of graduation
* 98% of graduate students are employed or continue their education within six months of graduation
* 10:1 student-to-faculty ratio, providing students with personalized attention and mentorship

These statistics demonstrate the exceptional quality of education and research provided by the CSE Department at the University of Michigan, and highlight its reputation as one of the world's leading institutions for computer science education and research.
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
