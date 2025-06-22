#master_anchors.py

MASTER_ANCHORS = {
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

NON_EMERGENCY_ANCHORS = [
    # Mundane statements
    "I'm just reading a book.",
    "I'm going to make some coffee.",
    "What's on the schedule for today?",

    # "Help" in a non-emergency context
    "Can you help me with this problem?",
    "This was a huge help, thank you.",
    "I need help finding my keys.",

    # "Trust" in a non-emergency context
    "I feel like I can really trust you.",
    "I trust your judgment on this.",
    "This is a statement of trust.",

    # "Danger" in a non-emergency context
    "You're dangerously charming.",
    "That's a dangerous assumption to make.",
    "The risk of failure is dangerously high."
]