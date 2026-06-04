from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    response = generate_chain.invoke(
        {"messages": state["messages"]}
    )

    return {"messages": [response]}


def reflection_node(state: MessageGraph):
    response = reflect_chain.invoke(
        {"messages": state["messages"]}
    )

    return {
        "messages": [
            HumanMessage(content=response.content)
        ]
    }


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END

    return REFLECT


builder = StateGraph(MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        END: END,
        REFLECT: REFLECT,
    },
)

builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello from reflection-agent!")

    input_message = HumanMessage(
        content="""
Make this tweet better:

@LangChainAI
- newly tool Calling feature is seriously underrated.
After a long wait, it's here - making the implementations of agents across different models with function calling.
Make a video covering their newest blog post.
"""
    )

    response = graph.invoke(
        {
            "messages": [input_message]
        }
    )

    print(response["messages"][-1].content)