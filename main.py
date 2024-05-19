from crewai import Crew, Process

from agents import YoutubeAutomationAgents
from tasks import YoutubeAutomationTasks
from langchain_openai import ChatOpenAI
from tools.youtube_video_details_tool import YoutubeVideoDetailsTool
from tools.youtube_video_search_tool import YoutubeVideoSearchTool

from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI GPT-4 language model
OpenAIGPT4 = ChatOpenAI()

agents = YoutubeAutomationAgents()
tasks = YoutubeAutomationTasks()

youtube_video_search_tool = YoutubeVideoSearchTool()
youtube_video_details_tool = YoutubeVideoDetailsTool()

youtube_manager = agents.youtube_manager()
research_manager = agents.research_manager(
    youtube_video_search_tool, youtube_video_details_tool
)
title_creator = agents.title_creator()
description_creator = agents.description_creator()
email_creator = agents.email_creator()

# TODO: UPDATE THE VIDEO DETAILS - The purpose of this video is to talk about how I've automated my YouTube video creation process using CrewAI, cover new CrewAI features, and how to build custom CrewAI tools

video_topic = "Best Developer Memes of 2024"
video_details = """
In this video, we're going to react to the best developer memes of 2024, explaining then and revisiting
the most viral discussions. This should be a fun video for Software developers.
"""

manage_youtube_video_creation = tasks.manage_youtube_video_creation(
    agent=youtube_manager, video_topic=video_topic, video_details=video_details
)
manage_youtube_video_research = tasks.manage_youtube_video_research(
    agent=research_manager,
    video_topic=video_topic,
    video_details=video_details,
)
create_youtube_video_title = tasks.create_youtube_video_title(
    agent=title_creator, video_topic=video_topic, video_details=video_details
)
create_youtube_video_description = tasks.create_youtube_video_description(
    agent=description_creator, video_topic=video_topic, video_details=video_details
)
create_email_announcement_for_new_video = tasks.create_email_announcement_for_new_video(
    agent=email_creator, video_topic=video_topic, video_details=video_details
)


# Create a new Crew instance
crew = Crew(
    agents=[
        youtube_manager,
        research_manager,
        email_creator,
    ],
    tasks=[
        manage_youtube_video_creation,
        manage_youtube_video_research,
        create_email_announcement_for_new_video,
    ],
    process=Process.hierarchical,
    manager_llm=OpenAIGPT4,
)

# Kick of the crew
results = crew.kickoff()

print("Crew usage", crew.usage_metrics)

print("Crew work results:")
print(results)
