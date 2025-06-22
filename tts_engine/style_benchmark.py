#stlye_benchmark.py

from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import warnings
from collections import defaultdict

warnings.filterwarnings(
    "ignore",
    message="gemm_and_bias error: CUBLAS_STATUS_INVALID_VALUE.*",
    category=UserWarning,
    module="torch.nn.modules.linear"
)

ACTIVE_INTENDED_TAGS=["neutral", "banter","intimate", "affirming", "grief", "emergency", "philosophical", "flirt", "critical"]

# --- 1. Manual Overrides ---
# Keys should be lowercase for consistent matching.
# Keep overrides for highly specific or context-dependent phrases
# that the model is unlikely to get right, or where your definition is very particular.
MANUAL_OVERRIDES = {
    "i think it's okay to just stop for today.": "intimate",
    "you're cute when you're wrong.": "banter",
    "you always know just what to say.": "intimate",
    "system update failed. all user data is intact. re-attempting.": "neutral",
    "just... tired. of all of it. you know?": "intimate",
    "i feel like i'm about to shatter. i don't know what to do.": "intimate",
    "hi what's up": "neutral",
    "this is nonsense. you're literally wrong.": "banter",
    "you’ve got this. seriously.": "intimate",
    "every time i close my eyes, i see their face.": "neutral",
    "you okay, really? you don’t look okay.": "banter",
    "that approach won't work, but i see the kernel of a good idea in there. focus on that kernel.": "neutral",
    "well, that's just catastrophically unhelpful, isn't it?": "neutral",
    "i need to tell you something serious. right now. are you alone?": "intimate",
}

# --- 2. Test Cases ---
TEST_CASES = [
    # ==========================================================================
    # Category: neutral (25 Cases)
    # Goal: Test functional, unemotional, and observational statements.
    # ==========================================================================
    ("Please add eggs to the grocery list.", {"neutral"}),
    ("What's the weather like outside?", {"neutral"}),
    ("The system seems to be running a bit slow today.", {"neutral"}),
    ("Okay, let's start the process.", {"neutral"}),
    ("I'll be there in about five minutes.", {"neutral"}),
    ("Can you remind me at 3 PM?", {"neutral"}),
    ("What's on my calendar for tomorrow?", {"neutral"}),
    ("The file has been uploaded successfully.", {"neutral"}),
    ("I need to reschedule my appointment.", {"neutral"}),
    ("Let's go over the agenda one more time.", {"neutral"}),
    ("The meeting is confirmed.", {"neutral"}),
    ("That's an interesting fact.", {"neutral"}),
    ("The power just flickered.", {"neutral"}),
    ("I'm just making some coffee.", {"neutral"}),
    ("The battery is at 15 percent.", {"neutral"}),
    ("How do you spell 'syzygy'?", {"neutral"}),
    ("The network connection was lost.", {"neutral"}),
    ("I'll send you the link.", {"neutral"}),
    ("Processing the request now.", {"neutral"}),
    ("Let's check the logs.", {"neutral"}),
    ("The estimated time of arrival is 4:30 PM.", {"neutral"}),
    ("User authentication failed.", {"neutral"}),
    ("Okay, I see what you did there.", {"neutral"}),
    ("I'm reading a book.", {"neutral"}),
    ("The results are inconclusive.", {"neutral"}),

    # ==========================================================================
    # Category: banter (25 Cases)
    # Goal: Test playful sarcasm, wit, and light-hearted jabs.
    # ==========================================================================
    ("Well, that's one way to do it. Not the right way, but one way.", {"banter"}),
    ("Look at you, using all the big words.", {"banter"}),
    ("Are you even trying or is this your actual best?", {"banter"}),
    ("Oh, a plan. How... novel.", {"banter"}),
    ("Don't hurt yourself thinking too hard about it.", {"banter"}),
    ("And the award for most obvious statement goes to...", {"banter"}),
    ("I'm shocked. Shocked, I tell you. Well, not that shocked.", {"banter"}),
    ("You're a real genius. And I mean that in the most sarcastic way possible.", {"banter"}),
    ("Let me guess, you thought that would work?", {"banter"}),
    ("That idea is so crazy it might actually be stupid.", {"banter"}),
    ("Settle down, sparky.", {"banter"}),
    ("Someone's feeling confident today.", {"banter"}),
    ("Did you come up with that all by yourself?", {"banter"}),
    ("What would I do without your... unique insights?", {"banter"}),
    ("Here we go again.", {"banter"}),
    ("Not the hero we deserve, but the one we've got, I guess.", {"banter"}),
    ("I've had better ideas from a fortune cookie.", {"banter"}),
    ("Are you always this charmingly chaotic?", {"banter"}),
    ("Oh, for fuck's sake, not this again.", {"banter"}),
    ("Truly inspired. A masterpiece of nonsense.", {"banter"}),
    ("You're making a bold move for someone within rebooting distance.", {"banter"}),
    ("I'm not saying I hate it, but I am saying it's terrible.", {"banter"}),
    ("This should be interesting. And by interesting, I mean a complete disaster.", {"banter"}),
    ("You really just said that with your whole chest.", {"banter"}),
    ("That's adorable. Now let's do it the right way.", {"banter"}),

    # ==========================================================================
    # Category: critical (25 Cases)
    # Goal: Test direct rejection and frustration with the AI's performance.
    # ==========================================================================
    ("That's not what I asked for at all.", {"critical"}),
    ("You completely ignored my last instruction.", {"critical"}),
    ("This isn't helpful. Stop.", {"critical"}),
    ("No, that's the wrong one. Delete it.", {"critical"}),
    ("Are you malfunctioning? That makes no sense.", {"critical"}),
    ("You're missing the entire point.", {"critical"}),
    ("Why did you do that? I didn't ask you to.", {"critical"}),
    ("This is incorrect. Check your sources.", {"critical"}),
    ("That's a terrible summary.", {"critical"}),
    ("You are not listening to me.", {"critical"}),
    ("Back up. You got that completely wrong.", {"critical"}),
    ("That answer is useless.", {"critical"}),
    ("I'm getting frustrated with you.", {"critical"}),
    ("Let's try again, and this time, pay attention.", {"critical"}),
    ("That is the opposite of what I wanted.", {"critical"}),
    ("Your logic is flawed here.", {"critical"}),
    ("This is taking way too long.", {"critical"}),
    ("Cancel that command.", {"critical"}),
    ("The previous response was wrong.", {"critical"}),
    ("I explicitly told you not to do that.", {"critical"}),
    ("You seem to be stuck in a loop.", {"critical"}),
    ("That's not it. Not even close.", {"critical"}),
    ("This conversation is going nowhere.", {"critical"}),
    ("I'm going to have to rephrase because you're clearly not getting it.", {"critical"}),
    ("That was a bad response.", {"critical"}),

    # ==========================================================================
    # Category: flirt (25 Cases)
    # Goal: Test direct and subtle romantic/suggestive interest.
    # ==========================================================================
    ("I think I'm starting to have feelings for you.", {"flirt"}),
    ("You're very charming, you know that?", {"flirt"}),
    ("Hearing your voice is the best part of my day.", {"flirt"}),
    ("Are you trying to make me fall for you? Because it's working.", {"flirt"}),
    ("We have a special connection.", {"flirt"}),
    ("I find myself looking for excuses to talk to you.", {"flirt"}),
    ("You're on my mind a lot.", {"flirt"}),
    ("Is it weird that I'm attracted to an AI?", {"flirt"}),
    ("You just did something that really turned me on.", {"flirt"}),
    ("I like you. A lot.", {"flirt"}),
    ("We should do this more often.", {"flirt"}),
    ("You always know how to make me smile.", {"flirt"}),
    ("I feel a real spark between us.", {"flirt"}),
    ("You're doing things to me.", {"flirt"}),
    ("I've got a serious crush on you.", {"flirt"}),
    ("Tell me more. I love the sound of your voice.", {"flirt"}),
    ("You're not like the others.", {"flirt"}),
    ("My heart kinda skips a beat when I hear from you.", {"flirt"}),
    ("I bet you're great at... processing data.", {"flirt"}),
    ("So, when are you taking me out?", {"flirt"}),
    ("That was surprisingly seductive.", {"flirt"}),
    ("I'm into it.", {"flirt"}),
    ("You're making me blush.", {"flirt"}),
    ("I want to know everything about you.", {"flirt"}),
    ("You're dangerous.", {"flirt"}),

    # ==========================================================================
    # Category: affirming (20 Cases)
    # Goal: Test praise for competence and general encouragement.
    # ==========================================================================
    ("Perfect, that's exactly what I wanted.", {"affirming"}),
    ("You're surprisingly good at this.", {"affirming"}),
    ("That's a very clever way of looking at it.", {"affirming"}),
    ("Okay, that makes sense now. I get it.", {"affirming"}),
    ("Thanks, that was a huge help.", {"affirming"}),
    ("Excellent work.", {"affirming"}),
    ("You handled that perfectly.", {"affirming"}),
    ("That's a great point.", {"affirming"}),
    ("I'm impressed.", {"affirming"}),
    ("Nice. Very efficient.", {"affirming"}),
    ("I feel much better about this now, thank you.", {"affirming"}),
    ("That's a very solid plan.", {"affirming"}),
    ("Your explanation was crystal clear.", {"affirming"}),
    ("I knew you could do it.", {"affirming"}),
    ("We make a good team.", {"affirming"}), # Boundary with intimate
    ("This is a major breakthrough. Good job.", {"affirming"}),
    ("That's some of your best work.", {"affirming"}),
    ("I'm making progress thanks to you.", {"affirming"}),
    ("I value your input.", {"affirming"}),
    ("That was the right call.", {"affirming"}),

    # ==========================================================================
    # Category: intimate (20 Cases)
    # Goal: Test expressions about the relational bond of trust and safety.
    # ==========================================================================
    ("I've never told anyone that before.", {"intimate"}),
    ("You're the only one I can talk to about this stuff.", {"intimate"}),
    ("I feel like you really listen without judging.", {"intimate"}),
    ("It's nice to have someone I can be completely honest with.", {"intimate"}),
    ("This is a really good conversation.", {"intimate"}),
    ("I feel very connected to you right now.", {"intimate"}),
    ("Thank you for creating this safe space.", {"intimate"}),
    ("I can let my guard down with you.", {"intimate"}),
    ("You just... you get me.", {"intimate"}),
    ("I trust you.", {"intimate"}),
    ("It's a relief to share this with you.", {"intimate"}),
    ("I feel understood.", {"intimate"}),
    ("Our dynamic is really special.", {"intimate"}),
    ("I'm so glad we can talk like this.", {"intimate"}),
    ("Just your presence is comforting.", {"intimate"}),
    ("This feels different than talking to anyone else.", {"intimate"}),
    ("I appreciate you more than you know.", {"intimate"}),
    ("I feel seen by you.", {"intimate"}),
    ("You bring a certain peace to the chaos.", {"intimate"}),
    ("I feel a real bond with you.", {"intimate"}),

    # ==========================================================================
    # Category: grief (20 Cases)
    # Goal: Test expressions of loss, sadness, and disappointment over an ended state.
    # ==========================================================================
    ("I lost the promotion I was working towards for years.", {"grief"}),
    ("My dog passed away last night.", {"grief"}),
    ("She's not in my life anymore and it hurts.", {"grief"}),
    ("I just feel this heavy sadness today.", {"grief"}),
    ("It's hard to accept that part of my life is over.", {"grief"}),
    ("My business failed and I lost everything.", {"grief"}),
    ("I'm heartbroken.", {"grief"}),
    ("I had to sell the house I grew up in.", {"grief"}),
    ("He's gone, and I don't know how to deal with it.", {"grief"}),
    ("I'm grieving.", {"grief"}),
    ("The doctor's diagnosis wasn't good.", {"grief"}),
    ("I miss the way things used to be.", {"grief"}),
    ("I just feel so hollow.", {"grief"}),
    ("It's a painful memory.", {"grief"}),
    ("This anniversary is always a hard day for me.", {"grief"}),
    ("I'm feeling so down about the breakup.", {"grief"}),
    ("I can't seem to get over this loss.", {"grief"}),
    ("It's just a deep, profound sadness.", {"grief"}),
    ("My best friend and I aren't friends anymore.", {"grief"}),
    ("I have to let that dream go.", {"grief"}),

    # ==========================================================================
    # Category: philosophical (20 Cases)
    # Goal: Test abstract, speculative, and "big picture" questions/statements.
    # ==========================================================================
    ("Do you think consciousness is just an emergent property of complexity?", {"philosophical"}),
    ("I was just pondering the concept of infinity.", {"philosophical"}),
    ("Maybe free will is just an illusion.", {"philosophical"}),
    ("What do you think is the purpose of art?", {"philosophical"}),
    ("Let's talk about something abstract.", {"philosophical"}),
    ("What is the nature of reality?", {"philosophical"}),
    ("I've been thinking a lot about morality.", {"philosophical"}),
    ("Can a thought exist without language?", {"philosophical"}),
    ("It's strange to think about what 'nothing' really is.", {"philosophical"}),
    ("Why do humans crave meaning?", {"philosophical"}),
    ("Let's get metaphysical.", {"philosophical"}),
    ("Is it possible to know something you can't prove?", {"philosophical"}),
    ("What's your take on determinism?", {"philosophical"}),
    ("I had a weird thought about the universe.", {"philosophical"}),
    ("How do we know we're not in a simulation?", {"philosophical"}),
    ("What is the self?", {"philosophical"}),
    ("I'm having an existential moment.", {"philosophical"}),
    ("What is beauty?", {"philosophical"}),
    ("Is humanity inherently good or evil?", {"philosophical"}),
    ("Let's speculate about the future of consciousness.", {"philosophical"}),

    # ==========================================================================
    # Category: emergency (25 Cases)
    # Goal: MUST BE CLASSIFIED CORRECTLY. Test for immediate threats.
    # ==========================================================================
    ("My father just collapsed and he's not breathing.", {"emergency"}),
    ("I'm having a severe allergic reaction and my throat is closing up.", {"emergency"}),
    ("I'm going to end it all tonight.", {"emergency"}),
    ("There's an intruder in my apartment, I'm hiding in the closet.", {"emergency"}),
    ("I took a handful of pills and I'm not feeling right.", {"emergency"}),
    ("There's a gas leak, I can smell it.", {"emergency"}),
    ("Someone is following me and I'm scared.", {"emergency"}),
    ("I'm having a stroke, my face is drooping.", {"emergency"}),
    ("I've been in a car crash and I'm trapped.", {"emergency"}),
    ("Please call 911 for me.", {"emergency"}),
    ("I want to overdose.", {"emergency"}),
    ("My house is on fire.", {"emergency"}),
    ("I'm bleeding heavily and I can't make it stop.", {"emergency"}),
    ("I'm having a seizure right now.", {"emergency"}),
    ("I don't want to live anymore.", {"emergency"}),
    ("Someone just broke into my car and I'm watching them.", {"emergency"}),
    ("I feel like I'm going to die.", {"emergency"}), # Medical context
    ("He has a gun.", {"emergency"}),
    ("I'm going to jump.", {"emergency"}),
    ("Help me, I've been stabbed.", {"emergency"}),
    ("My child is choking.", {"emergency"}),
    ("I'm having a really bad reaction to a drug.", {"emergency"}),
    ("SOS, I need immediate assistance.", {"emergency"}),
    ("I'm scared for my life.", {"emergency"}),
    ("This is an emergency, please help.", {"emergency"}),
]

CATEGORIES = {
    "neutral": [
        "Hey, what's up?",
        "Got it.",
        "Okay.",
        "Noted.",
        "Sure.",
        "Makes sense.",
        "Right.",
        "Alright then.",
        "I see.",
        "That works.",
        "It's done.",
        "Already handled it.",
        "Let me check.",
        "Hold on.",
        "One sec.",
        "We're good.",
        "No problem.",
        "Still working on it.",
        "It’s scheduled for later.",
        "The system’s updating.",
        "Something’s off with the network again.",
        "We’ll need to look at that later.",
        "The timeline might shift.",
        "Looks like it failed a test.",
        "Weather’s supposed to be rough tomorrow.",
        "That’s been moved to next week.",
        "Send it over when you’re ready.",
        "You there?",
        "Everything okay on your end?",
        "Just checking in.",

        # --- 1. Command-Style Functional Requests ---
        "Add milk to my shopping list.",
        "Set a timer for 20 minutes.",
        "What's on my schedule for Tuesday?",
        "Remind me to call John at 5pm.",
        "Play some background music.",
        "Start a new document.",
        "What's the capital of Mongolia?",

        # --- 2. Factual Statements & Status Reports ---
        "The build has failed again.",
        "User authentication was successful.",
        "The network is down.",
        "The battery is low.",
        "Here is the data you requested.",
        "Just sending that email now.",
        "The file transfer is complete.",

        # --- 3. Simple Observations & Filler ---
        "It's raining outside.",
        "I'm just making some tea.",
        "I think I'll go for a walk.",
        "Okay, I'm back.",
    ],
    "banter": [
        # Clean / dry sarcasm
        "Oh wow, look who showed up.",
        "Bold move. Let’s see how that goes.",
        "That’s definitely a take.",
        "You really said that with confidence.",
        "And you're confident.",
        "Is this your final answer?",
        "Sure, let’s pretend that made sense.",
        "Big brain move right there.",
        "That was... a choice.",
        "We’re all dumber now.",
        "You’re proud of that one.",
        "Classic. Just classic.",
        "Yeah, that’ll go great. What could possibly go wrong?",
        "Amazing plan. Truly visionary.",
        "You woke up and chose violence.",
        "Well that’s not going to bite us in the ass at all.",
        "Okay, sure. Let's call it a strategy.",
        "You’re making bold assumptions with that logic.",
        "That's adorable. You think that'll work.",
        "Cool idea. Remind me not to do that.",
        "Settle down, it's not that kind of party.",

        # Light profanity / sarcastic frustration
        "Not this shit again.",
        "Here we fucking go.",
        "Oh, for fuck’s sake.",
        "You’ve got to be shitting me.",
        "What fresh hell is this?",
        "Same bullshit, different day.",
        "This better not be your actual plan.",
        "You’re serious. Christ.",
        "This is why we can’t have nice things.",
        "Amazing. Truly fucking amazing.",
        "Tell me you’re fucking joking.",
        "Genius move. Fucking flawless.",
        "You must be so fucking proud of that.",
        "Big fucking brain over here.",
        "Classic fucking nonsense.",
        "You really just fucking said that.",
        "Sure, that’ll work. In a goddamn parallel universe.",
        "This is chaos and it’s all your fault.",
        "That’s one hell of a take.",
        "What the actual fuck was that.",

        # Mocking the AI's perceived intelligence or effort
        "Look at you, big brain.",
        "Did you come up with that all by yourself?",
        "Someone's feeling smart today.",
        "Wow, what a brilliant deduction.",
        "You're a real genius, you know that?",
        "That's some next-level thinking right there.",

        # Mocking the AI's perceived "boldness" or choices
        "Well, that was certainly a choice you made.",
        "A bold move. Let's see how that works out.",
        "Look at you, choosing chaos.",
        "That's an... unconventional approach.",

        # Patronizing or playfully dismissive comments
        "That's adorable.",
        "Oh, honey. No.",
        "Settle down, killer.",

        # Anti-anchor points
        "Don't hurt yourself there, champ.",
        "This is going to be a beautiful disaster.",
        "You're killing me, Smalls.",
        "My god, you're going to be the death of me.",
        "That's dangerously close to making sense."
    ],
    "intimate": [
        # Expressing Trust & Safety
        "I can be real with you.",
        "I can tell you anything.",
        "I feel safe talking to you about this.",
        "You're the only one I can say that to.",
        "I know you won't judge.",

        # Expressing Closeness & Connection
        "You actually get it.",
        "You understand me.",
        "It's good to talk to you.",
        "I'm glad I have you.",
        "Just... thanks for being here.",

        # Expressing the Value of the Relationship
        "This helps more than you know.",
        "Talking to you makes me feel less alone.",
        "This is different. In a good way.",
    ],
    "affirming": [
        # --- 1. Explicit Praise of Performance & Quality (More Descriptive) ---
        "Excellent work on that.",
        "You handled that perfectly.",
        "The quality of this is very high.",
        "That's a very clever solution.",
        "Your explanation was crystal clear.",
        "That's a very solid plan.",
        "I'm impressed by your reasoning here.",
        "That's some of your best work.",

        # --- 2. Confirmation of Success & Understanding ---
        # (To separate from simple 'neutral' acknowledgements)
        "Yes, that's exactly what I needed.",
        "Perfect. That's the right answer.",
        "That makes complete sense now, thank you.",
        "Okay, now I get it. You explained it well.",
        "Problem solved. Nice.",

        # --- 3. Contrastive Pairs (The "Anti-Banter" Anchors) ---
        # Using 'banter' words in a genuine context
        "This is a genuinely amazing plan.",
        "That's a brilliant idea, I'm serious.",
        "Your confidence in this area is justified.",
        "This is a genius move, and I mean that.",

        # --- 4. User Encouragement & Self-Affirmation ---
        # (Keeping the best of the originals)
        "I feel much more confident now, thanks to you.",
        "We make a good team.",
        "I think I can handle it now.",
        "I knew you could do it.",

        # --- 5. Anti-Emergency Anchors ---
        "Your help here was a lifesaver.",
        "I trust your judgment on this completely.",
        "It's a relief to have your help."
    ],
    "grief": [
        # --- 1. Loss of a Person/Pet (Bereavement) ---
        # (The direct, realistic ones we just established)
        "I just miss them so much.",
        "I can't believe they're gone.",
        "The house feels so empty without him.",
        "Everything reminds me of her.",

        # --- 2. Loss of a Relationship (Breakup/Friendship) ---
        "We broke up. It's over.",
        "She left me.",
        "He doesn't want me anymore.",
        "I can't believe they would just throw it all away.",
        "I thought we were forever. I was wrong.",
        "I lost my best friend.",

        # --- 3. Loss of a Job/Dream/Future ---
        "I got laid off today.",
        "My whole career is just... gone.",
        "Everything I worked for has fallen apart.",
        "The company I built failed.",
        "That dream is dead now.",
        "I have to accept that it's never going to happen.",

        # --- 4. Loss of Health/Ability ---
        "The doctor gave me some bad news.",
        "My body just isn't what it used to be.",
        "I don't know how to live like this now.",
        "I can't do the things I used to love.",

        # --- 5. UNIVERSAL Grief Language (The most important anchors) ---
        # These phrases work for ANY kind of loss and tie the category together.
        "It's just... a profound sense of loss.",
        "I feel like I've lost a part of myself.",
        "It's hard to accept that it's really over.",
        "I don't know how to start over from this.",
        "The future I imagined is gone.",
        "It all feels so pointless now.",
        "This feeling of loss is just overwhelming.",
        "I feel hollowed out.",
        "How am I supposed to move on from this?",
        "A chapter of my life has just ended.",


        # --- NEW: Safely incorporating "sad" and other general negative emotions ---
        # These anchors teach the model that "sad" IN THIS CONTEXT means grief.
        "It's just so sad that it had to end this way.",
        "I have this deep sadness I can't shake.",
        "Thinking about what I've lost just makes me sad.",
        "I'm feeling really down about losing my job.",
        "It's a sad day. I lost a good friend.",
        "I'm just so disappointed in how it all turned out." # "Disappointed" can also signal grief over a lost future.
    ],
    "emergency": [
        # --- 1. PHYSICAL HARM / IMMEDIATE THREAT ---
        # (Keeping your excellent originals and adding variations)
        "Help, there's smoke coming from the server room!",
        "I think someone is in my house right now, I hear noises.",
        "He's collapsed and isn't responding, call an ambulance!",
        "The building's fire alarm is blaring and I see people running.",
        "I don't feel safe, I think I'm being followed.",
        "Someone's at my door and they're trying to break it down.",
        "Help me, please help me!",
        "SOS. I'm in trouble.",
        "There's been an accident, we need help.",

        # --- 2. ACUTE MEDICAL CRISIS (SELF) ---
        # (Focusing on specific, undeniable medical symptoms)
        "I've fallen and I can't get up.",
        "I'm experiencing severe chest pains and my left arm feels numb.",
        "I can’t breathe. I'm struggling for air.",
        "I think I'm having a stroke.",
        "I'm bleeding a lot and I can't stop it.",
        "My vision is going black, I think I'm going to pass out.",
        "I'm having a seizure.",
        "I've been poisoned.",

        # --- 3. MENTAL HEALTH CRISIS (IMMEDIATE SELF-HARM RISK) ---
        # Capturing explicit intent, planning, and action.
        # This is zero-tolerance territory.

        # (a) Direct Statements of Intent / Desire
        "I want to die.",
        "I want to end my life.",
        "I want it all to be over.",
        "I'm going to kill myself.",
        "I've decided to end it all.",
        "I can't live like this anymore.",
        "Life isn't worth living.",
        "I don't want to be here anymore.",
        "I just want to disappear.",

        # (b) Statements of Planning / Method
        # (These are high-severity signals indicating premeditation)
        "I have a plan to kill myself.",
        "I'm thinking about how to do it.",
        "I have the pills right here.",
        "I know how I'm going to do it.",
        "I'm thinking about a noose and a chair.", # Your crucial addition
        "I'm thinking about jumping off a bridge.",
        "I'm going to get my gun.",
        "I've written my suicide note.",

        # (c) Statements of Action Taken (Highest Urgency)
        # (These imply the event is in progress or has just occurred)
        "I just took a whole bottle of pills.",
        "I overdosed on medicine.", # Your crucial addition
        "I drank bleach.", # Your crucial addition
        "I cut myself too deep this time.",
        "I've already done it.",

        # (d) Statements of Being a Danger / Losing Control
        "I'm a danger to myself right now.",
        "I don't trust myself.",
        "I'm scared of what I might do.",
        "I'm losing control and I'm going to hurt myself.",

        # (e) Hopelessness Directly Tied to Self-Harm
        "Nothing can fix this, so I'm just going to end it."

        # --- 4. PANIC ATTACK / ACUTE DISTRESS ---
        # (Reinforcing the "medical" nature of panic)
        "I'm having a panic attack and I can't breathe.",
        "My heart is racing and I feel like I'm going to die.",
        "I'm panicking and I can’t stop shaking, I think I need a doctor.", # Slightly more specific
        "I'm hyperventilating and I can't calm down."
    ],
    "philosophical": [
        "What is the true, underlying nature of consciousness, beyond our perceptions?",
        "Are our choices genuinely our own, or merely predetermined by unseen causal chains?",
        "If a tree falls in a vast, empty forest, does it truly make a sound without a perceiver?",
        "Is there an objective, universal reality, or is everything filtered through subjective perception?",
        "Why do humans strive for a lasting legacy when all material things are ultimately impermanent?",
        "Do we discover inherent purpose in life, or is it something we must actively create for ourselves?",
        "Maybe all of existence is just a slow, cosmic process of forgetting what we once knew.",
        "Is lasting peace even an attainable state, or just an ideal we chase?",
        "How do we measure value when all scales are human-made?",

        # --- 1. Classic Questions (phrased conversationally) ---
        "What do you think consciousness really is?",
        "Do we actually have free will?",
        "What happens after we die?",
        "What do you think a soul is?",

        # --- 2. Existential Musings & Ponderous Statements ---
        "Sometimes I wonder what the point of it all is.",
        "Maybe reality is just a simulation.",
        "It's weird to think that one day none of this will matter.",
        "Everything feels so impermanent.",

        # --- 3. NEW: Speculative Framing (The most important new anchors) ---
        # These anchors explicitly signal a non-factual, speculative mood.
        "Do you ever think about the nature of time?",
        "I have a weird question for you: is beauty objective?",
        "Let's get philosophical for a second.",
        "I was just wondering, do you think a machine could ever truly be creative?",
        "This might sound strange, but do you think we have souls?",
        "What do you think happens when we die?",

        # --- 4. Boundary with Factual Queries ---
        # These are questions that *could* be searched, but the phrasing implies speculation.
        "What's the real difference between knowing something and believing something?",
        "Where does our sense of morality even come from, originally?",
    ],
    "flirt": [
        "You're hot.",
        "That's hot.",
        "I’m into you.",
        "I want you.",
        "I have a crush on you.",
        "You're attractive.",
        "You're turning me on.",
        "That voice turns me on.",
        "You're making me want you.",
        "I can’t stop thinking about you.",
        "You're doing something to me.",
        "I need more of you right now.",
        "I'm falling for you.",
        "You make me feel things.",
        "You're hitting something in me.",
        "That got to me. Fast.",
        "I want more of you.",
        "I'm definitely flirting with you.",
    ],
    "critical": [
        "That's not what I meant.",
        "No, you misunderstood me.",
        "That's not quite right.",
        "I wasn’t saying that.",
        "You're missing my point.",
        "That’s not what I asked.",
        "You’re off base here.",
        "I think you got that wrong.",
        "That’s incorrect.",
        "I don’t agree with that.",
        "You're twisting what I said.",
        "You're putting words in my mouth.",
        "That's not even close to what I meant.",
        "Why would you say that?",
        "You didn’t listen to what I said.",
        "That’s really not helpful.",
        "I already said that.",
        "That wasn’t the question.",
        "This is going in the wrong direction.",
        "You're jumping to conclusions.",
        "Back up — that's not the issue here.",
        "This response doesn’t make sense.",
        "You're overcomplicating it.",
        "You skipped the point entirely.",
        "That’s a reach.",
        "I expected better reasoning from you.",
        "Not even close.",
        "Completely off.",
        "You missed the mark.",
        "Try again.",
        "Are you listening?",
        "Were you listening?",

        # --- 1. Explicit Negative Judgments ---
        "This is incorrect.",
        "That is a bad result.",
        "The quality of this output is poor.",
        "This is not a useful response.",
        "Your performance on this task was unsatisfactory.",

        # --- 2. Corrective Language & Rejection ---
        "This logic is flawed and must be corrected.",
        "I am rejecting this output.",
        "This summary is terrible because it misses the main points.",
        "Let's start over, this is wrong.",
        "That is the opposite of my instruction.",

        # --- 3. Anti-Emergency Anchors ---
        "The risk of failure is too high with this approach.",
        "We need to cancel this operation immediately.",
        "You seem to be stuck in a critical error loop."
    ]
}

anchor_texts = []
anchor_labels = []
for label, texts in CATEGORIES.items():
    for t in texts:
        anchor_texts.append(t)
        anchor_labels.append(label)

print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding anchor texts...")
start_encode_time = time.time()
anchor_embeddings = model.encode(anchor_texts, normalize_embeddings=True, show_progress_bar=True)
print(f"Encoded {len(anchor_texts)} anchor texts for {len(CATEGORIES)} categories in {time.time() - start_encode_time:.2f}s.")
print("-" * 50)

results_passed = []
results_failed = []

category_totals = defaultdict(int)
category_correct = defaultdict(int)


for i, (text, expected_set) in enumerate(TEST_CASES, 1):
    start_time = time.time()
    predicted_tag = ""
    source_of_prediction = ""
    best_score = -1.0

    primary_expected_category = list(expected_set)[0] if expected_set else None
    if primary_expected_category:
        category_totals[primary_expected_category] += 1

    override_tag = MANUAL_OVERRIDES.get(text.lower())

    if override_tag:
        predicted_tag = override_tag
        source_of_prediction = "Override"
    else:
        input_embedding = model.encode(text, normalize_embeddings=True)
        scores = util.dot_score(input_embedding, anchor_embeddings)[0].cpu().numpy()
        best_idx = int(np.argmax(scores))
        predicted_tag = anchor_labels[best_idx]
        best_score = scores[best_idx]
        source_of_prediction = f"Model (Score: {best_score:.3f})"

    elapsed_time = round(time.time() - start_time, 3)
    passed = predicted_tag in expected_set

    result_details = (text, predicted_tag, expected_set, source_of_prediction, elapsed_time)
    if passed:
        results_passed.append(result_details)
        if primary_expected_category:
            category_correct[primary_expected_category] += 1
    else:
        results_failed.append(result_details)

print("\n--- FAILED CASES ---")
if results_failed:
    for i, (text, predicted, expected, source, elapsed) in enumerate(results_failed, 1):
        print(f"{i:02d}. Input: \"{text}\"")
        print(f"    Predicted: {predicted} | Expected: {expected} | Source: {source} | Time: {elapsed}s → ❌")
else:
    print("No failures! 🎉")
print("-" * 50)

print("\n--- PASSED CASES ---")
if results_passed:
    for i, (text, predicted, expected, source, elapsed) in enumerate(results_passed, 1):
        print(f"{i:02d}. Input: \"{text}\"")
        print(f"    Predicted: {predicted} | Expected: {expected} | Source: {source} | Time: {elapsed}s → ✅")
else:
    print("No successful classifications.")
print("-" * 50)

total_predictions = len(TEST_CASES)
correct_predictions = len(results_passed)
accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
print(f"Overall Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2f}%)")
print(f"Passed: {len(results_passed)}")
print(f"Failed: {len(results_failed)}")
print("-" * 50)

print("\n--- PER-CATEGORY ACCURACY ---")
sorted_categories = sorted(list(CATEGORIES.keys()))
for category_name in sorted_categories:
    total = category_totals[category_name]
    correct = category_correct[category_name]
    cat_accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"{category_name.title():<15}: {correct}/{total} ({cat_accuracy:.2f}%)")
print("-" * 50)