"""SGLang frontend program: gen, gen with choices, and fork.

Start the server first:
  python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000
Then: python sglang_program.py        (pip install sglang)
"""
from sglang import function, system, user, assistant, gen, RuntimeEndpoint
from sglang.lang.api import set_default_backend


@function
def review(s, text):
    s += system("You are a short, clear reviewer.")
    s += user("Review this text: " + text)
    s += assistant(gen("review", max_tokens=60))
    s += user("Is it good or bad? One word.")
    s += assistant(gen("verdict", choices=["good", "bad"]))


@function
def two_ideas(s, topic):
    s += user("Give a project idea about " + topic)
    forks = s.fork(2)  # two branches that share the prompt prefix in the radix cache
    for f in forks:
        f += assistant(gen("idea", max_tokens=40, temperature=0.9))
    s += assistant("Ideas: " + forks[0]["idea"] + " | " + forks[1]["idea"])


if __name__ == "__main__":
    set_default_backend(RuntimeEndpoint("http://localhost:30000"))
    state = review(text="The movie was slow but the ending was great.")
    print(state["review"], "->", state["verdict"])
    state = two_ideas(topic="LLM serving")
    print(state.text())
