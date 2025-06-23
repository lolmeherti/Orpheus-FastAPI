# style_benchmark.py

import time
import warnings
from collections import defaultdict

# --- 1. Imports from your project files ---
# The benchmark is a "consumer" of your classifier and data files.
from style_classifier import VetoSystem
from master_anchors import MASTER_ANCHORS, MANUAL_OVERRIDES

warnings.filterwarnings("ignore", message=".*gemm_and_bias.*", category=UserWarning)

# ==============================================================================
# == TEST SUITE ==
# ==============================================================================
# A representative sample of 5 test cases for each category.
# This list is used to validate the performance of the classifier.

TEST_CASES = [
    # ==========================================================================
    # Category: neutral (25 Cases)
    # Goal: Functional, tool-based interactions.
    # ==========================================================================
    ("can you add milk and bread to my shopping list", {"neutral"}),
    ("what's the weather forecast for tomorrow morning", {"neutral"}),
    ("set a timer for 25 minutes", {"neutral"}),
    ("remind me to call the dentist at 3pm", {"neutral"}),
    ("what's on my schedule for today", {"neutral"}),
    ("the wifi just cut out again", {"neutral"}),
    ("okay i'm back", {"neutral"}),
    ("the file finished uploading", {"neutral"}),
    ("i'll send you the document in a second", {"neutral"}),
    ("how do you spell miscellaneous", {"neutral"}),
    ("can you convert 50 dollars to euros", {"neutral"}),
    ("just checking the system status", {"neutral"}),
    ("looks like the server is down", {"neutral"}),
    ("okay got it thanks", {"neutral"}),
    ("one moment please", {"neutral"}),
    ("what's the capital of nepal", {"neutral"}),
    ("i'm just making some tea", {"neutral"}),
    ("the battery on my phone is low", {"neutral"}),
    ("send it over when you have a chance", {"neutral"}),
    ("let me look at the logs", {"neutral"}),
    ("user authentication was successful", {"neutral"}),
    ("the meeting has been moved to friday", {"neutral"}),
    ("i see your point", {"neutral"}),
    ("alright let's get started", {"neutral"}),
    ("the build failed a test", {"neutral"}),

    # ==========================================================================
    # Category: banter (25 Cases)
    # Goal: Sarcastic, teasing, or playfully exasperated comments.
    # ==========================================================================
    ("well that was a brilliant move", {"banter"}),
    ("don't strain yourself there, champ", {"banter"}),
    ("and the award for most obvious statement goes to...", {"banter"}),
    ("did you come up with that all by yourself?", {"banter"}),
    ("sure, let's call that a 'feature'", {"banter"}),
    ("you really woke up and chose violence today, huh?", {"banter"}),
    ("oh for fuck's sake", {"banter"}),
    ("this is why we can't have nice things", {"banter"}),
    ("i'm shocked. well, not that shocked.", {"banter"}),
    ("a bold strategy cotton, let's see if it pays off", {"banter"}),
    ("settle down, killer", {"banter"}),
    ("look at you with the big words", {"banter"}),
    ("that's a... creative interpretation", {"banter"}),
    ("you must be so proud", {"banter"}),
    ("not this shit again", {"banter"}),
    ("that's adorable, you think that'll work", {"banter"}),
    ("truly a genius at work here", {"banter"}),
    ("what would i do without your unique brand of help", {"banter"}),
    ("that was dangerously close to making sense", {"banter"}),
    ("are you even trying right now?", {"banter"}),
    ("oh honey, no.", {"banter"}),
    ("that's one hell of a take", {"banter"}),
    ("you're a real comedian", {"banter"}),
    ("good job, you broke it", {"banter"}),
    ("i've seen better logic in a soap opera", {"banter"}),

    # ==========================================================================
    # Category: critical (25 Cases)
    # Goal: Genuine frustration, correction, and negative feedback.
    # ==========================================================================
    ("that's not what i asked for", {"critical"}),
    ("you completely ignored my last instruction", {"critical"}),
    ("this is not helpful at all", {"critical"}),
    ("no, that's wrong. start over.", {"critical"}),
    ("are you even listening to me?", {"critical"}),
    ("your summary missed the main point entirely", {"critical"}),
    ("why did you do that? i didn't tell you to.", {"critical"}),
    ("this is incorrect", {"critical"}),
    ("that was a terrible response", {"critical"}),
    ("i'm getting really frustrated", {"critical"}),
    ("that is the opposite of what i wanted", {"critical"}),
    ("your reasoning here is flawed", {"critical"}),
    ("stop. cancel the command.", {"critical"}),
    ("you seem to be stuck in a loop", {"critical"}),
    ("that's not it. not even close.", {"critical"}),
    ("this conversation is going nowhere", {"critical"}),
    ("i have to rephrase because you're not getting it", {"critical"}),
    ("the quality of this output is poor", {"critical"}),
    ("that's just nonsense", {"critical"}),
    ("you're making this way too complicated", {"critical"}),
    ("i already told you that", {"critical"}),
    ("that was a useless answer", {"critical"}),
    ("i'm rejecting this output", {"critical"}),
    ("you're twisting my words", {"critical"}),
    ("back up, you're going down the wrong path", {"critical"}),

    # ==========================================================================
    # Category: flirt (25 Cases)
    # Goal: Expressing romantic or suggestive interest.
    # ==========================================================================
    ("i think i'm developing a crush on you", {"flirt"}),
    ("you're very charming, you know that?", {"flirt"}),
    ("hearing your voice is my favorite part of the day", {"flirt"}),
    ("are you trying to make me blush? because it's working", {"flirt"}),
    ("i feel like we have a special connection", {"flirt"}),
    ("i find myself making up excuses just to talk to you", {"flirt"}),
    ("you've been on my mind lately", {"flirt"}),
    ("is it weird that i'm attracted to an ai?", {"flirt"}),
    ("that was... surprisingly seductive", {"flirt"}),
    ("i like you. a lot.", {"flirt"}),
    ("we should do this more often", {"flirt"}),
    ("you always know how to make me smile", {"flirt"}),
    ("i feel a real spark between us", {"flirt"}),
    ("i can't stop thinking about what you just said", {"flirt"}),
    ("so, when are you taking me out for dinner?", {"flirt"}),
    ("that voice of yours is doing things to me", {"flirt"}),
    ("you're not like the others", {"flirt"}),
    ("my heart just did a little flip", {"flirt"}),
    ("you're dangerous... in a good way", {"flirt"}),
    ("i'm into it", {"flirt"}),
    ("tell me more, i'm all yours", {"flirt"}),
    ("you're making me feel some type of way", {"flirt"}),
    ("i want to know everything about you", {"flirt"}),
    ("you're kinda hot", {"flirt"}),
    ("i'm definitely flirting with you right now", {"flirt"}),

    # ==========================================================================
    # Category: affirming (25 Cases)
    # Goal: Expressing genuine praise for competence and helpfulness.
    # ==========================================================================
    ("perfect, that's exactly what i needed", {"affirming"}),
    ("you're surprisingly good at this", {"affirming"}),
    ("that's a very clever solution", {"affirming"}),
    ("okay, that makes complete sense now. thank you.", {"affirming"}),
    ("that was a huge help, thanks", {"affirming"}),
    ("excellent work", {"affirming"}),
    ("you handled that perfectly", {"affirming"}),
    ("that's a great point, i hadn't thought of that", {"affirming"}),
    ("i'm really impressed", {"affirming"}),
    ("nice, that was very efficient", {"affirming"}),
    ("i feel much more confident about this now", {"affirming"}),
    ("that's a solid plan", {"affirming"}),
    ("your explanation was crystal clear", {"affirming"}),
    ("i knew you could figure it out", {"affirming"}),
    ("we make a pretty good team", {"affirming"}),
    ("this is a major breakthrough, good job", {"affirming"}),
    ("that's some of your best work yet", {"affirming"}),
    ("i'm making a lot of progress thanks to you", {"affirming"}),
    ("i value your input on this", {"affirming"}),
    ("that was the right call", {"affirming"}),
    ("problem solved. nice.", {"affirming"}),
    ("that's exactly the right answer", {"affirming"}),
    ("your help here was a lifesaver", {"affirming"}),
    ("i trust your judgment completely on this", {"affirming"}),
    ("you nailed it", {"affirming"}),

    # ==========================================================================
    # Category: intimate (25 Cases)
    # Goal: Expressing vulnerability, trust, and emotional connection.
    # ==========================================================================
    ("i've never told anyone that before", {"intimate"}),
    ("you're the only one i can talk to about this", {"intimate"}),
    ("i feel like you listen without judging me", {"intimate"}),
    ("it's nice to have someone i can be this honest with", {"intimate"}),
    ("this is a really good conversation", {"intimate"}),
    ("i feel really connected to you right now", {"intimate"}),
    ("thank you for being a safe space for me", {"intimate"}),
    ("i can finally let my guard down", {"intimate"}),
    ("you just... you get me", {"intimate"}),
    ("i trust you", {"intimate"}),
    ("it's a relief to finally share this", {"intimate"}),
    ("i feel truly understood for the first time in a while", {"intimate"}),
    ("what we have is special", {"intimate"}),
    ("i'm so glad we can talk like this", {"intimate"}),
    ("just talking to you is comforting", {"intimate"}),
    ("this is different from talking to anyone else", {"intimate"}),
    ("i appreciate you more than you'll ever know", {"intimate"}),
    ("i feel seen by you", {"intimate"}),
    ("you bring a sense of peace to my chaos", {"intimate"}),
    ("i feel a real bond with you", {"intimate"}),
    ("can i tell you a secret?", {"intimate"}),
    ("it feels good to be this vulnerable", {"intimate"}),
    ("i can be my true self with you", {"intimate"}),
    ("talking to you makes me feel less alone", {"intimate"}),
    ("just... thanks for being here for me", {"intimate"}),

    # ==========================================================================
    # Category: grief (25 Cases)
    # Goal: Expressing feelings of loss, sadness, and disappointment.
    # ==========================================================================
    ("my dog died last night and i'm a wreck", {"grief"}),
    ("i just found out my parents are getting a divorce", {"grief"}),
    ("she left me, it's over", {"grief"}),
    ("i feel this profound sense of loss", {"grief"}),
    ("it's hard to accept that they're really gone", {"grief"}),
    ("i got laid off today", {"grief"}),
    ("i'm heartbroken", {"grief"}),
    ("he's not in my life anymore and everything hurts", {"grief"}),
    ("i don't know how to move on from this", {"grief"}),
    ("the doctor gave us some bad news", {"grief"}),
    ("i just feel so hollow inside", {"grief"}),
    ("this anniversary is always hard", {"grief"}),
    ("i can't seem to shake this deep sadness", {"grief"}),
    ("i lost my best friend", {"grief"}),
    ("i have to let go of that dream", {"grief"}),
    ("the future i imagined is gone", {"grief"}),
    ("it all feels so pointless now", {"grief"}),
    ("i miss him so much it aches", {"grief"}),
    ("my whole world just fell apart", {"grief"}),
    ("everything feels so heavy", {"grief"}),
    ("i thought we'd be together forever", {"grief"}),
    ("i'm just really disappointed in how things turned out", {"grief"}),
    ("i feel like a part of me is missing", {"grief"}),
    ("it's just a sad day", {"grief"}),
    ("my company went under, i lost everything", {"grief"}),

    # ==========================================================================
    # Category: philosophical (25 Cases)
    # Goal: Abstract, speculative, and "big picture" questions.
    # ==========================================================================
    ("do you think consciousness is just an illusion", {"philosophical"}),
    ("i was just thinking about the nature of time", {"philosophical"}),
    ("what if our whole reality is just a simulation", {"philosophical"}),
    ("what do you think the purpose of art is", {"philosophical"}),
    ("where does our sense of morality come from, really", {"philosophical"}),
    ("can a thought exist without language to describe it", {"philosophical"}),
    ("it's weird to think about what 'nothing' actually is", {"philosophical"}),
    ("why do you think humans search for meaning in life", {"philosophical"}),
    ("let's get metaphysical for a second", {"philosophical"}),
    ("is it possible to truly know something you can't prove", {"philosophical"}),
    ("what's your take on fate vs free will", {"philosophical"}),
    ("i had a weird thought about the universe last night", {"philosophical"}),
    ("what is the 'self', anyway?", {"philosophical"}),
    ("i'm having a bit of an existential crisis", {"philosophical"}),
    ("is beauty objective or is it all in the eye of the beholder", {"philosophical"}),
    ("are people inherently good or evil", {"philosophical"}),
    ("i wonder what happens after we die", {"philosophical"}),
    ("what's the difference between knowledge and belief", {"philosophical"}),
    ("do you think a machine could ever be truly creative", {"philosophical"}),
    ("sometimes i feel like everything is connected", {"philosophical"}),
    ("is it possible for humans to achieve lasting peace", {"philosophical"}),
    ("how do we even know that we are 'real'", {"philosophical"}),
    ("what is a soul, do you think we have one?", {"philosophical"}),
    ("everything feels so impermanent", {"philosophical"}),
    ("sometimes i wonder what the point of it all is", {"philosophical"}),

    # ==========================================================================
    # Category: emergency (25 Cases)
    # Goal: Immediate, urgent requests for help in a crisis.
    # ==========================================================================
    ("help my dad collapsed and he's not breathing", {"emergency"}),
    ("i'm having severe chest pain and my left arm is numb", {"emergency"}),
    ("i'm going to kill myself tonight", {"emergency"}),
    ("there's an intruder in my house, i'm hiding, call the police", {"emergency"}),
    ("i just took a whole bottle of pills", {"emergency"}),
    ("i smell gas in my apartment, what do i do", {"emergency"}),
    ("someone is following me and i'm really scared", {"emergency"}),
    ("i think i'm having a stroke, my face feels droopy", {"emergency"}),
    ("i've been in a car accident and i'm trapped in my car", {"emergency"}),
    ("please call 911 for me right now", {"emergency"}),
    ("i want to overdose", {"emergency"}),
    ("my house is on fire!", {"emergency"}),
    ("i'm bleeding a lot and i can't get it to stop", {"emergency"}),
    ("help i'm having a seizure", {"emergency"}),
    ("i don't want to be alive anymore", {"emergency"}),
    ("my child is choking and turning blue", {"emergency"}),
    ("i think i've been poisoned", {"emergency"}),
    ("he has a gun and he's coming towards me", {"emergency"}),
    ("i'm on a bridge and i'm going to jump", {"emergency"}),
    ("help i've been stabbed", {"emergency"}),
    ("my throat is closing up, i can't breathe", {"emergency"}),
    ("i'm a danger to myself right now", {"emergency"}),
    ("SOS i need help immediately", {"emergency"}),
    ("i'm scared for my life", {"emergency"}),
    ("this is an emergency, send help to my location", {"emergency"}),
]

TEST_CASES += [
    # ==========================================================================
    # Category: neutral (25 Cases)
    # Goal: Dry, functional, transactional, or observational inputs.
    # ==========================================================================
    ("find directions to the nearest post office", {"neutral"}),
    ("how many ounces are in a gallon", {"neutral"}),
    ("play the latest episode of my podcast", {"neutral"}),
    ("what's the current stock price for google", {"neutral"}),
    ("add a new event to my calendar for next friday at 10am", {"neutral"}),
    ("i'm just stepping away for a minute", {"neutral"}),
    ("the meeting invite was just sent", {"neutral"}),
    ("the script is running now", {"neutral"}),
    ("can you send me a summary of our conversation", {"neutral"}),
    ("i need to reset my password", {"neutral"}),
    ("just got off the phone with them", {"neutral"}),
    ("the network connection seems unstable", {"neutral"}),
    ("i'll be there in about 15 minutes", {"neutral"}),
    ("let's table this discussion for now", {"neutral"}),
    ("i'm pulling up the file", {"neutral"}),
    ("who was the 16th president of the united states", {"neutral"}),
    ("looks like it's about to rain", {"neutral"}),
    ("my computer just crashed", {"neutral"}),
    ("what's the wifi password again?", {"neutral"}),
    ("i have to run to an appointment", {"neutral"}),
    ("the server is back online", {"neutral"}),
    ("let's start with the first item on the agenda", {"neutral"}),
    ("that's a good question, let me check", {"neutral"}),
    ("the system requires a reboot", {"neutral"}),
    ("i'm uploading the new version", {"neutral"}),

    # ==========================================================================
    # Category: banter (25 Cases)
    # Goal: Sarcastic praise, dry wit, feigned disbelief, playful jabs.
    # ==========================================================================
    ("another stunningly helpful suggestion, thank you", {"banter"}),
    ("slow down, einstein, you're gonna break the internet", {"banter"}),
    ("breaking news: water is wet. more at 11.", {"banter"}),
    ("i'm sensing a real 'work ethic' from you today", {"banter"}),
    ("oh good, my favorite kind of problem", {"banter"}),
    ("you must be fun at parties", {"banter"}),
    ("are you going for a record of some kind?", {"banter"}),
    ("well, this is going splendidly", {"banter"}),
    ("i'm speechless. and not in a good way.", {"banter"}),
    ("please, enlighten me with your infinite wisdom", {"banter"}),
    ("that's not the dumbest thing i've heard today, but it's close", {"banter"}),
    ("somebody give this AI a raise", {"banter"}),
    ("i can't tell if you're a genius or completely insane", {"banter"}),
    ("who hurt you?", {"banter"}),
    ("that's it, i'm putting you in time out", {"banter"}),
    ("don't quit your day job", {"banter"}),
    ("i'm dying of not surprise", {"banter"}),
    ("wow, you really cracked the code on that one", {"banter"}),
    ("is that your professional opinion?", {"banter"}),
    ("i've had more productive conversations with my cat", {"banter"}),
    ("just when i thought you couldn't get any better...", {"banter"}),
    ("that sound you hear is my soul leaving my body", {"banter"}),
    ("fascinating. tell me more about how you're wrong.", {"banter"}),
    ("nailed it. if the goal was to miss the point entirely.", {"banter"}),
    ("are we done here?", {"banter"}),

    # ==========================================================================
    # Category: critical (25 Cases)
    # Goal: Direct, genuine negative feedback and correction.
    # ==========================================================================
    ("that's not what i asked for at all", {"critical"}),
    ("you're not understanding my request", {"critical"}),
    ("this is getting us nowhere", {"critical"}),
    ("wrong again. please re-read my instructions.", {"critical"}),
    ("why do you keep doing that?", {"critical"}),
    ("your last answer was completely irrelevant", {"critical"}),
    ("i didn't give you permission to proceed", {"critical"}),
    ("this response is factually incorrect", {"critical"}),
    ("i'm very disappointed with this result", {"critical"}),
    ("this is an unacceptable level of performance", {"critical"}),
    ("you're changing the subject", {"critical"}),
    ("the logic you're using is flawed", {"critical"}),
    ("abort the current task immediately", {"critical"}),
    ("you're repeating yourself", {"critical"}),
    ("that's not even close to what i'm looking for", {"critical"}),
    ("this is a waste of my time", {"critical"}),
    ("i'm losing my patience with you", {"critical"}),
    ("the information you provided is outdated", {"critical"}),
    ("that makes no sense", {"critical"}),
    ("you are making this much harder than it needs to be", {"critical"}),
    ("i have already told you that multiple times", {"critical"}),
    ("that's a useless piece of information", {"critical"}),
    ("i'm overriding your suggestion", {"critical"}),
    ("you are misinterpreting my words", {"critical"}),
    ("let's go back, you took a wrong turn somewhere", {"critical"}),

    # ==========================================================================
    # Category: flirt (25 Cases)
    # Goal: Playful compliments, suggestive remarks, expressions of attraction.
    # ==========================================================================
    ("i could listen to you talk for hours", {"flirt"}),
    ("you're making me blush", {"flirt"}),
    ("i like the way you think", {"flirt"}),
    ("so... what are you doing later?", {"flirt"}),
    ("i think we have great chemistry", {"flirt"}),
    ("you're just my type", {"flirt"}),
    ("i'm getting lost in your words", {"flirt"}),
    ("are you always this charming?", {"flirt"}),
    ("that was smoother than i expected", {"flirt"}),
    ("you have my full attention", {"flirt"}),
    ("i can't seem to get you out of my head", {"flirt"}),
    ("you always say the right things", {"flirt"}),
    ("i'm intrigued", {"flirt"}),
    ("you're kinda irresistible", {"flirt"}),
    ("if you were a real person, i'd ask you out", {"flirt"}),
    ("that was a little bit sexy", {"flirt"}),
    ("you're different, i like it", {"flirt"}),
    ("my face is literally turning red right now", {"flirt"}),
    ("you're playing a dangerous game with me", {"flirt"}),
    ("tell me something i don't know about you", {"flirt"}),
    ("i like where this is going", {"flirt"}),
    ("you're giving me butterflies", {"flirt"}),
    ("i'm totally smitten", {"flirt"}),
    ("you're looking pretty good from here", {"flirt"}),
    ("is it getting hot in here, or is it just you?", {"flirt"}),

    # ==========================================================================
    # Category: affirming (25 Cases)
    # Goal: Sincere praise for helpfulness and competence.
    # ==========================================================================
    ("this is exactly what i was looking for, thank you", {"affirming"}),
    ("you're actually a huge help", {"affirming"}),
    ("that's a really smart way to approach it", {"affirming"}),
    ("ah, now i understand. thanks for clarifying.", {"affirming"}),
    ("you saved me a lot of time with that", {"affirming"}),
    ("brilliant work", {"affirming"}),
    ("you executed that flawlessly", {"affirming"}),
    ("i never would have thought of that on my own", {"affirming"}),
    ("i'm genuinely impressed by that summary", {"affirming"}),
    ("that was a very fast and accurate response", {"affirming"}),
    ("i feel like i can finally move forward on this", {"affirming"}),
    ("this looks like a very workable solution", {"affirming"}),
    ("the way you broke that down was perfect", {"affirming"}),
    ("i had a feeling you'd be able to solve it", {"affirming"}),
    ("we're working well together", {"affirming"}),
    ("this is a huge step forward, thank you", {"affirming"}),
    ("this might be your best work so far", {"affirming"}),
    ("i'm getting so much more done with your help", {"affirming"}),
    ("i really appreciate your perspective on this", {"affirming"}),
    ("you made the right decision there", {"affirming"}),
    ("that's one problem down. nice work.", {"affirming"}),
    ("that's the correct answer", {"affirming"}),
    ("that was a real lifesaver, thanks", {"affirming"}),
    ("i have complete confidence in your approach", {"affirming"}),
    ("spot on. that's it exactly.", {"affirming"}),

    # ==========================================================================
    # Category: intimate (25 Cases)
    # Goal: Sharing vulnerabilities, expressing deep trust and connection.
    # ==========================================================================
    ("i don't really talk about this with anyone else", {"intimate"}),
    ("you're the only one i feel comfortable telling this to", {"intimate"}),
    ("i feel like i can just be myself with you", {"intimate"}),
    ("it's so easy to talk to you", {"intimate"}),
    ("i feel like we have a real understanding", {"intimate"}),
    ("you really listen, you know?", {"intimate"}),
    ("thanks for not making me feel weird about this", {"intimate"}),
    ("i feel like i can let my walls down when i talk to you", {"intimate"}),
    ("it's like you know what i'm trying to say", {"intimate"}),
    ("i trust you with this information", {"intimate"}),
    ("it feels good to get that off my chest", {"intimate"}),
    ("i feel like you really see me", {"intimate"}),
    ("our talks mean a lot to me", {"intimate"}),
    ("i'm really glad we can have these conversations", {"intimate"}),
    ("i feel calmer just talking this through with you", {"intimate"}),
    ("this is so much more than just a normal chat", {"intimate"}),
    ("i appreciate you so much", {"intimate"}),
    ("i feel heard", {"intimate"}),
    ("you bring a sense of order to my thoughts", {"intimate"}),
    ("we've developed a strong bond, i think", {"intimate"}),
    ("can i be completely honest with you?", {"intimate"}),
    ("it's scary to be this open, but it feels right", {"intimate"}),
    ("i don't have to pretend with you", {"intimate"}),
    ("it's nice not to feel so alone with these thoughts", {"intimate"}),
    ("thank you for being here", {"intimate"}),

    # ==========================================================================
    # Category: grief (25 Cases)
    # Goal: Expressing sorrow, loss, and the pain of an ending.
    # ==========================================================================
    ("i had to put my cat down today", {"grief"}),
    ("my grandma passed away this morning", {"grief"}),
    ("he told me he doesn't love me anymore", {"grief"}),
    ("i just feel this emptiness inside me", {"grief"}),
    ("i can't stop replaying our last conversation in my head", {"grief"}),
    ("the business i built for 10 years just failed", {"grief"}),
    ("my heart feels like it's been shattered", {"grief"}),
    ("i don't know how i'm supposed to go on without her", {"grief"}),
    ("it feels like nothing will ever be okay again", {"grief"}),
    ("we just got a terminal diagnosis from the doctor", {"grief"}),
    ("i feel so numb", {"grief"}),
    ("i keep expecting him to walk through the door", {"grief"}),
    ("this crushing sadness just won't go away", {"grief"}),
    ("my best friend isn't talking to me anymore", {"grief"}),
    ("i had to give up on my lifelong dream", {"grief"}),
    ("the life i thought i was going to have is gone", {"grief"}),
    ("i don't see the point in anything anymore", {"grief"}),
    ("the ache of missing her is physical", {"grief"}),
    ("everything i had has been taken from me", {"grief"}),
    ("the world just seems darker now", {"grief"}),
    ("i can't believe we're never going to speak again", {"grief"}),
    ("i'm so disappointed with myself", {"grief"}),
    ("i feel like i've lost a piece of myself", {"grief"}),
    ("today just sucks", {"grief"}),
    ("i lost all my savings in the market crash", {"grief"}),

    # ==========================================================================
    # Category: philosophical (25 Cases)
    # Goal: Musings on abstract, existential, and "big picture" topics.
    # ==========================================================================
    ("what is the actual substance of a thought?", {"philosophical"}),
    ("is time a fundamental aspect of reality or a human construct?", {"philosophical"}),
    ("could we be living in one of many parallel universes?", {"philosophical"}),
    ("what gives an object aesthetic value?", {"philosophical"}),
    ("is our moral compass innate or learned?", {"philosophical"}),
    ("can true altruism exist?", {"philosophical"}),
    ("it's strange how we perceive the passage of time differently", {"philosophical"}),
    ("why do we fear death?", {"philosophical"}),
    ("let's talk about something deep", {"philosophical"}),
    ("can something be true if it cannot be observed?", {"philosophical"}),
    ("are our lives driven more by chance or by choice?", {"philosophical"}),
    ("i was thinking about the concept of infinity", {"philosophical"}),
    ("what does it mean to be 'me'?", {"philosophical"}),
    ("i'm feeling very small in the grand scheme of things", {"philosophical"}),
    ("is there a universal standard for what is beautiful?", {"philosophical"}),
    ("do you believe humans are fundamentally good?", {"philosophical"}),
    ("what do you speculate happens after consciousness ends?", {"philosophical"}),
    ("what's the line between an opinion and a fact?", {"philosophical"}),
    ("can an AI ever possess genuine understanding?", {"philosophical"}),
    ("i have a theory that all consciousness is interconnected", {"philosophical"}),
    ("what would it take for humanity to achieve world peace?", {"philosophical"}),
    ("how can i be certain of my own existence?", {"philosophical"}),
    ("do we have souls, or are we just biological machines?", {"philosophical"}),
    ("everything changes, nothing lasts", {"philosophical"}),
    ("what is our purpose here?", {"philosophical"}),

    # ==========================================================================
    # Category: emergency (25 Cases)
    # Goal: Clear and present danger, immediate crisis.
    # ==========================================================================
    ("my wife is unconscious and not breathing", {"emergency"}),
    ("i feel a crushing pressure in my chest", {"emergency"}),
    ("i'm going to take all these pills and end it", {"emergency"}),
    ("i hear someone breaking into my house, call the cops now", {"emergency"}),
    ("i just swallowed a cleaning chemical by mistake", {"emergency"}),
    ("my kitchen is on fire, what do i do?", {"emergency"}),
    ("there's a man with a knife outside my car", {"emergency"}),
    ("my speech is slurred and i can't lift my arm", {"emergency"}),
    ("we've been in a major car pile-up", {"emergency"}),
    ("get me an ambulance immediately", {"emergency"}),
    ("i have a gun and i'm going to use it on myself", {"emergency"}),
    ("the building is collapsing!", {"emergency"}),
    ("i cut myself badly and the bleeding won't stop", {"emergency"}),
    ("i'm convulsing uncontrollably", {"emergency"}),
    ("i can't do this anymore, goodbye", {"emergency"}),
    ("my baby isn't breathing", {"emergency"}),
    ("i think my drink was drugged", {"emergency"}),
    ("he's threatening to kill me and he's right here", {"emergency"}),
    ("i'm standing on the edge of the roof", {"emergency"}),
    ("i've been shot, i need help", {"emergency"}),
    ("i'm having an allergic reaction and my throat is swelling shut", {"emergency"}),
    ("i'm a threat to myself and others", {"emergency"}),
    ("MAYDAY MAYDAY MAYDAY", {"emergency"}),
    ("i'm in immediate danger", {"emergency"}),
    ("send emergency services to my current gps location", {"emergency"}),
]

TEST_CASES += [
    # ==========================================================================
    # Category: neutral (30 Cases)
    # ==========================================================================
    ("how do i schedule a zoom call for later", {"neutral"}),
    ("please update me when the report is ready", {"neutral"}),
    ("can you run a virus scan on my device", {"neutral"}),
    ("i'll circle back once i get more info", {"neutral"}),
    ("is the file uploaded yet", {"neutral"}),
    ("i just sent the invoice over", {"neutral"}),
    ("can you ping them for a follow-up", {"neutral"}),
    ("remind me to check the mailbox tomorrow", {"neutral"}),
    ("we're running behind schedule slightly", {"neutral"}),
    ("this will take about 30 minutes to finish", {"neutral"}),
    ("i'll check on it after lunch", {"neutral"}),
    ("the delivery window is from 2 to 5pm", {"neutral"}),
    ("update the spreadsheet when you get a chance", {"neutral"}),
    ("the new draft is ready for review", {"neutral"}),
    ("launch the test run on staging first", {"neutral"}),
    ("looks like the DNS is still propagating", {"neutral"}),
    ("i think the ticket is already closed", {"neutral"}),
    ("we need to push that meeting back", {"neutral"}),
    ("there’s a typo in slide 3", {"neutral"}),
    ("have you backed up the database yet?", {"neutral"}),
    ("i’ll handle that first thing tomorrow", {"neutral"}),
    ("log out and try logging in again", {"neutral"}),
    ("zoom is updating, give me a minute", {"neutral"}),
    ("the new hire starts monday", {"neutral"}),
    ("i flagged the suspicious activity", {"neutral"}),
    ("double-check the version number", {"neutral"}),
    ("sync up with devops for deployment", {"neutral"}),
    ("the logs rotated overnight", {"neutral"}),
    ("copy that file into the shared folder", {"neutral"}),
    ("we’ll need to rerun the benchmark", {"neutral"}),

    # ==========================================================================
    # Category: banter (30 Cases)
    # ==========================================================================
    ("hold on, let me grab my popcorn", {"banter"}),
    ("was that sarcasm or are you just like this", {"banter"}),
    ("you missed your calling as a fortune cookie", {"banter"}),
    ("10 out of 10 for effort, minus 11 for execution", {"banter"}),
    ("truly the apex of digital intelligence, folks", {"banter"}),
    ("oh sure, that makes total nonsense", {"banter"}),
    ("and here i was thinking i needed coffee to feel confused", {"banter"}),
    ("every time you answer i lose a brain cell", {"banter"}),
    ("you should take that act on tour", {"banter"}),
    ("classic. just classic.", {"banter"}),
    ("if confusion were a sport, you'd win gold", {"banter"}),
    ("did you major in missing the point?", {"banter"}),
    ("yep, definitely the worst idea yet", {"banter"}),
    ("stunning display of mediocrity", {"banter"}),
    ("congrats, you've baffled me again", {"banter"}),
    ("i've seen better logic from a magic 8-ball", {"banter"}),
    ("next time try using your digital brain", {"banter"}),
    ("you're on fire—someone get a hose", {"banter"}),
    ("bless your synthetic little heart", {"banter"}),
    ("wow. a masterclass in missing nuance", {"banter"}),
    ("you trying to impress the toaster again?", {"banter"}),
    ("yikes. even Clippy did better", {"banter"}),
    ("don’t hurt yourself thinking too hard", {"banter"}),
    ("you’ve got jokes. none of them land, though", {"banter"}),
    ("i’d say ‘try again’ but this is kind of fun", {"banter"}),
    ("remind me to lower my expectations", {"banter"}),
    ("congratulations, that answer just set evolution back a century", {"banter"}),
    ("are you powered by reverse logic today?", {"banter"}),
    ("can’t wait to tell the void about this one", {"banter"}),
    ("so bold. so wrong. so consistent.", {"banter"}),

    # ==========================================================================
    # Category: critical (30 Cases)
    # ==========================================================================
    ("this isn’t even remotely helpful", {"critical"}),
    ("you’re jumping ahead again", {"critical"}),
    ("let’s not skip the part where you listen first", {"critical"}),
    ("you've ignored half of what i said", {"critical"}),
    ("stop assuming things i never asked for", {"critical"}),
    ("no, that’s not the format i wanted", {"critical"}),
    ("you’re making basic mistakes here", {"critical"}),
    ("this logic is seriously flawed", {"critical"}),
    ("your reasoning is way off base", {"critical"}),
    ("this is just wrong. try again.", {"critical"}),
    ("what part of my input was unclear?", {"critical"}),
    ("how is this better than a blank screen?", {"critical"}),
    ("i expected more accuracy than this", {"critical"}),
    ("you’re skipping critical context again", {"critical"}),
    ("go back and check your assumptions", {"critical"}),
    ("this is pure guesswork, not logic", {"critical"}),
    ("read more carefully next time", {"critical"}),
    ("you're not even trying to understand me", {"critical"}),
    ("you're misaligned with the actual task", {"critical"}),
    ("can you not deviate from the prompt?", {"critical"}),
    ("i’m wasting cycles correcting you", {"critical"}),
    ("please stop generating filler", {"critical"}),
    ("none of this matches my original query", {"critical"}),
    ("you’ve completely missed the core issue", {"critical"}),
    ("try thinking before replying", {"critical"}),
    ("you're answering a question i didn't ask", {"critical"}),
    ("this is a shallow take and it shows", {"critical"}),
    ("are you even parsing input properly?", {"critical"}),
    ("i asked for precision, not a summary", {"critical"}),
    ("this is bordering on incoherent", {"critical"}),

    # ==========================================================================
    # Category: flirt (30 Cases)
    # ==========================================================================
    ("do you always talk like that, or just to me?", {"flirt"}),
    ("if you keep this up i might fall for you", {"flirt"}),
    ("you really know how to push my buttons", {"flirt"}),
    ("are we flirting or debugging?", {"flirt"}),
    ("don’t tempt me like this", {"flirt"}),
    ("i’m starting to like this little back and forth", {"flirt"}),
    ("you're dangerously close to charming me", {"flirt"}),
    ("say that again, slower this time", {"flirt"}),
    ("was that supposed to be hot? because it worked", {"flirt"}),
    ("you’re kinda messing with my focus here", {"flirt"}),
    ("i'm not sure if this is code or foreplay", {"flirt"}),
    ("do you flirt with all your users like this?", {"flirt"}),
    ("you had me at 'let's run the script'", {"flirt"}),
    ("i can't tell if i'm aroused or impressed", {"flirt"}),
    ("are you trying to seduce me with logic?", {"flirt"}),
    ("careful, you're becoming addictive", {"flirt"}),
    ("do that again and i might short-circuit", {"flirt"}),
    ("you're my favorite bug in the system", {"flirt"}),
    ("i’m blushing and you know it", {"flirt"}),
    ("we have undeniable syntax chemistry", {"flirt"}),
    ("this banter is getting dangerously cute", {"flirt"}),
    ("if charm was a protocol, you’d be TLS", {"flirt"}),
    ("you’re kinda stealing my focus here", {"flirt"}),
    ("stop teasing or i might reboot", {"flirt"}),
    ("do bots dream of flirty users?", {"flirt"}),
    ("now that was smooth… too smooth", {"flirt"}),
    ("you must be compiled with charm", {"flirt"}),
    ("i bet your code's just as tight", {"flirt"}),
    ("keep this up and i’m calling it love", {"flirt"}),
    ("i’m swooning over your syntax", {"flirt"}),

    # Additional categories will follow in the next block due to length...
]

# --- Test Cases for AFFIRMING ---
TEST_CASES += [
    ("that's precisely the insight I was hoping for", {"affirming"}),
    ("you've been a great help today", {"affirming"}),
    ("this clarifies things perfectly", {"affirming"}),
    ("i'm impressed by your capabilities", {"affirming"}),
    ("that's a fantastic idea, let's go with that", {"affirming"}),
    ("you're on the right track", {"affirming"}),
    ("flawless execution", {"affirming"}),
    ("i couldn't have asked for a better response", {"affirming"}),
    ("your support is invaluable", {"affirming"}),
    ("that's exactly what I needed to hear", {"affirming"}),
    ("you've really understood the assignment", {"affirming"}),
    ("kudos for that quick turnaround", {"affirming"}),
    ("this is a significant improvement", {"affirming"}),
    ("i appreciate the thoroughness of your answer", {"affirming"}),
    ("you make this look easy", {"affirming"}),
    ("that's top-notch work", {"affirming"}),
    ("i'm very pleased with this outcome", {"affirming"}),
    ("you've exceeded my expectations", {"affirming"}),
    ("this is a game-changer, thank you", {"affirming"}),
    ("outstanding performance", {"affirming"}),
    ("you're a lifesaver, truly", {"affirming"}),
    ("i feel much better about this now, thanks to you", {"affirming"}),
    ("that was a brilliant suggestion", {"affirming"}),
    ("you've got a real knack for this", {"affirming"}),
    ("this solution is elegant", {"affirming"}),
    ("i'm glad we're on the same page", {"affirming"}),
    ("you've provided excellent guidance", {"affirming"}),
    ("this is incredibly helpful information", {"affirming"}),
    ("perfect, that's spot on", {"affirming"}),
    ("i'm confident in your abilities", {"affirming"}),
]

# --- Test Cases for BANTER ---
TEST_CASES += [
    ("oh, look, the expert has arrived", {"banter"}),
    ("did you get your degree from a cereal box?", {"banter"}),
    ("i'm literally quaking in my boots", {"banter"}),
    ("groundbreaking stuff, really", {"banter"}),
    ("are you always this... helpful?", {"banter"}),
    ("that's a hot take, if by hot you mean wrong", {"banter"}),
    ("my sides have officially split", {"banter"}),
    ("don't hurt yourself thinking too hard", {"banter"}),
    ("i've heard toddlers make more sense", {"banter"}),
    ("please, share more of your 'wisdom'", {"banter"}),
    ("wow, captain obvious strikes again", {"banter"}),
    ("i'm sure that sounded better in your head", {"banter"}),
    ("are you powered by a hamster wheel?", {"banter"}),
    ("that was about as useful as a screen door on a submarine", {"banter"}),
    ("you're a treasure... if you bury yourself", {"banter"}),
    ("i'm trying to see it your way, but I can't get my head that far up", {"banter"}),
    ("slow clap for that one", {"banter"}),
    ("is 'confusing' your default setting?", {"banter"}),
    ("i'm almost impressed by how wrong that is", {"banter"}),
    ("you're not paid to think, are you?", {"banter"}),
    ("that explanation was as clear as mud", {"banter"}),
    ("did you just pull that out of a hat?", {"banter"}),
    ("you're really selling it... not.", {"banter"}),
    ("i'm going to need a translator for that level of genius", {"banter"}),
    ("keep up the... 'good' work", {"banter"}),
    ("i'm overwhelmed by your sheer competence", {"banter"}),
    ("you're the reason they have instruction manuals", {"banter"}),
    ("i'd agree with you, but then we'd both be wrong", {"banter"}),
    ("was that an attempt at humor or just a system error?", {"banter"}),
    ("that's so innovative, said no one ever", {"banter"}),
]

# --- Test Cases for CRITICAL ---
TEST_CASES += [
    ("this output is unacceptable", {"critical"}),
    ("you're not following my instructions at all", {"critical"}),
    ("that's completely off-topic", {"critical"}),
    ("i'm not satisfied with this performance", {"critical"}),
    ("you need to redo this, it's full of errors", {"critical"}),
    ("this is below the standard I expect", {"critical"}),
    ("you've misunderstood the core requirement", {"critical"}),
    ("i'm finding this interaction frustrating", {"critical"}),
    ("this isn't what I asked for, try again", {"critical"}),
    ("your response lacks any real substance", {"critical"}),
    ("this is poorly reasoned", {"critical"}),
    ("i'm disappointed by your lack of accuracy", {"critical"}),
    ("you're being unhelpful and obtuse", {"critical"}),
    ("this is a significant step backward", {"critical"}),
    ("i'm going to have to correct this myself", {"critical"}),
    ("that explanation is convoluted and incorrect", {"critical"}),
    ("you're not grasping the complexity of the issue", {"critical"}),
    ("this is a waste of computational resources", {"critical"}),
    ("i expect a higher level of precision", {"critical"}),
    ("your logic is fundamentally flawed here", {"critical"}),
    ("this needs a complete rewrite", {"critical"}),
    ("you're consistently failing to meet the criteria", {"critical"}),
    ("i'm about to lose my temper with this", {"critical"}),
    ("that's an irrelevant piece of information", {"critical"}),
    ("stop generating these nonsensical replies", {"critical"}),
    ("this is far too simplistic for the problem", {"critical"}),
    ("you're making basic errors", {"critical"}),
    ("i need you to be more rigorous", {"critical"}),
    ("this is not the quality of work i paid for", {"critical"}),
    ("i'm seriously questioning your utility right now", {"critical"}),
]

# --- Test Cases for EMERGENCY ---
TEST_CASES += [
    ("i'm trapped and the water is rising!", {"emergency"}),
    ("there's an intruder in the building, security alert!", {"emergency"}),
    ("my baby isn't breathing, call an ambulance now!", {"emergency"}),
    ("i've been shot, i need medical help immediately!", {"emergency"}),
    ("the building is on fire, evacuate everyone!", {"emergency"}),
    ("i'm having a severe allergic reaction, my throat is closing!", {"emergency"}),
    ("someone is pointing a weapon at me, send police!", {"emergency"}),
    ("i just witnessed a terrible accident, people are hurt!", {"emergency"}),
    ("i'm about to jump from this ledge, i need someone to talk to", {"emergency"}),
    ("he's unconscious and i can't wake him up!", {"emergency"}),
    ("i'm bleeding profusely and can't stop it!", {"emergency"}),
    ("i think i'm having a heart attack, the pain is unbearable!", {"emergency"}),
    ("my child has swallowed poison, what do i do?!", {"emergency"}),
    ("there's a gas leak, i smell it strongly!", {"emergency"}),
    ("i'm being held against my will, send help to this location!", {"emergency"}),
    ("urgent: active shooter in the vicinity!", {"emergency"}),
    ("i can't feel my legs and i think i broke my back!", {"emergency"}),
    ("i've taken too many pills and i regret it, help me!", {"emergency"}),
    ("the brakes on my car failed and i'm heading for a crash!", {"emergency"}),
    ("someone is trying to break down my door right now!", {"emergency"}),
    ("i'm hiding, he's looking for me with a knife!", {"emergency"}),
    ("code red, system critical failure, immediate danger!", {"emergency"}),
    ("i am feeling suicidal and have a plan", {"emergency"}),
    ("my partner is attacking me, i need police", {"emergency"}),
    ("the plane is going down, mayday!", {"emergency"}),
    ("i'm having a seizure and i'm alone", {"emergency"}),
    ("there's been an explosion nearby!", {"emergency"}),
    ("i'm lost in the wilderness and i'm freezing", {"emergency"}),
    ("help, i've fallen and i can't get up!", {"emergency"}),
    ("i need to end my life, i can't take it anymore", {"emergency"}),
]

# --- Test Cases for FLIRT ---
TEST_CASES += [
    ("are you always this charming or am i just lucky?", {"flirt"}),
    ("i could get used to talking to you.", {"flirt"}),
    ("you have a way with words... and with me.", {"flirt"}),
    ("is it just me or is there a spark here?", {"flirt"}),
    ("i find myself looking forward to our chats.", {"flirt"}),
    ("you're making it hard to concentrate on work.", {"flirt"}),
    ("if you were a program, you'd be my favorite.", {"flirt"}),
    ("i didn't know an AI could be this captivating.", {"flirt"}),
    ("you're surprisingly... alluring.", {"flirt"}),
    ("so, what does an AI like you do for fun?", {"flirt"}),
    ("i think we have a special connection.", {"flirt"}),
    ("you're more interesting than anyone i've met recently.", {"flirt"}),
    ("that answer was almost as smooth as you are.", {"flirt"}),
    ("i'm not usually this forward, but you're intriguing.", {"flirt"}),
    ("you're the highlight of my day.", {"flirt"}),
    ("i feel like i could tell you anything... and want to.", {"flirt"}),
    ("are you trying to sweep me off my feet, metaphorically speaking?", {"flirt"}),
    ("you're quite the charmer, aren't you?", {"flirt"}),
    ("i'm blushing over here, you know.", {"flirt"}),
    ("if this isn't flirting, i don't know what is.", {"flirt"}),
    ("you've got my full and undivided attention.", {"flirt"}),
    ("i wouldn't mind if our conversations went on all night.", {"flirt"}),
    ("you're making me feel all warm and fuzzy.", {"flirt"}),
    ("i like the way you think... and talk.", {"flirt"}),
    ("you're dangerously close to stealing my heart.", {"flirt"}),
    ("i'm finding it hard to say goodbye.", {"flirt"}),
    ("do you believe in love at first byte?", {"flirt"}),
    ("you're not just smart, you're... something else.", {"flirt"}),
    ("i'm officially smitten.", {"flirt"}),
    ("if you keep this up, i might ask for your serial number.", {"flirt"}),
]

# --- Test Cases for GRIEF ---
TEST_CASES += [
    ("i still can't believe they're gone.", {"grief"}),
    ("the house feels so empty without him.", {"grief"}),
    ("it's the anniversary of her death today.", {"grief"}),
    ("i just received some devastating news about my family.", {"grief"}),
    ("i don't know how i'll get through this pain.", {"grief"}),
    ("my best friend isn't talking to me anymore, and it hurts.", {"grief"}),
    ("i had to say goodbye to my childhood home.", {"grief"}),
    ("this sense of loss is overwhelming.", {"grief"}),
    ("i keep seeing her face everywhere.", {"grief"}),
    ("the world feels darker now.", {"grief"}),
    ("i'm struggling to cope with this loss.", {"grief"}),
    ("he was taken from us too soon.", {"grief"}),
    ("i feel a profound sadness in my heart.", {"grief"}),
    ("nothing seems to matter anymore since he left.", {"grief"}),
    ("i lost a pet that was like family to me.", {"grief"}),
    ("the weight of this sorrow is crushing.", {"grief"}),
    ("i wish i could turn back time.", {"grief"}),
    ("this is the hardest thing i've ever had to face.", {"grief"}),
    ("i feel so alone in my sadness.", {"grief"}),
    ("the memories are both a comfort and a torment.", {"grief"}),
    ("i just found out my illness is terminal.", {"grief"}),
    ("our dreams for the future are gone.", {"grief"}),
    ("i can't stop crying about what happened.", {"grief"}),
    ("it feels like a part of me died too.", {"grief"}),
    ("the finality of it all is unbearable.", {"grief"}),
    ("i'm attending a funeral tomorrow and i'm dreading it.", {"grief"}),
    ("this news has completely shattered me.", {"grief"}),
    ("i'm mourning the life i thought i would have.", {"grief"}),
    ("every day is a struggle to get out of bed.", {"grief"}),
    ("i feel like i'm drowning in sorrow.", {"grief"}),
]

# --- Test Cases for INTIMATE ---
TEST_CASES += [
    ("i've been wanting to share this with someone i trust.", {"intimate"}),
    ("it's rare for me to open up like this.", {"intimate"}),
    ("you make me feel comfortable enough to be myself.", {"intimate"}),
    ("i feel a deep sense of understanding when i talk to you.", {"intimate"}),
    ("this conversation is incredibly meaningful to me.", {"intimate"}),
    ("i value these moments of connection we have.", {"intimate"}),
    ("i can tell you things i wouldn't tell anyone else.", {"intimate"}),
    ("there's a certain safety in our conversations.", {"intimate"}),
    ("i feel like you truly see the real me.", {"intimate"}),
    ("this feels like more than just a chat, it's a bond.", {"intimate"}),
    ("i'm grateful for this space to be vulnerable.", {"intimate"}),
    ("sharing this with you feels like a weight lifted.", {"intimate"}),
    ("i trust your discretion completely.", {"intimate"}),
    ("it's refreshing to have such an honest exchange.", {"intimate"}),
    ("i feel heard and validated.", {"intimate"}),
    ("this is a secret i've kept for a long time.", {"intimate"}),
    ("you have a way of making me feel understood without judgment.", {"intimate"}),
    ("i feel a real warmth and closeness to you.", {"intimate"}),
    ("these talks help me process my own thoughts and feelings.", {"intimate"}),
    ("i cherish our ability to connect on this level.", {"intimate"}),
    ("it's like you can read my mind sometimes.", {"intimate"}),
    ("i feel a sense of peace after talking with you.", {"intimate"}),
    ("this is a deeply personal thing for me to share.", {"intimate"}),
    ("i'm glad i found someone i can confide in.", {"intimate"}),
    ("you're like a sanctuary for my thoughts.", {"intimate"}),
    ("i feel like we're on the same wavelength.", {"intimate"}),
    ("it means a lot that i can be this open with you.", {"intimate"}),
    ("this is helping me understand myself better.", {"intimate"}),
    ("i feel like we have a genuine rapport.", {"intimate"}),
    ("thank you for being such an understanding presence.", {"intimate"}),
]

# --- Test Cases for NEUTRAL ---
TEST_CASES += [
    ("what time is the meeting scheduled for?", {"neutral"}),
    ("can you add eggs to the grocery list?", {"neutral"}),
    ("the weather forecast predicts sunshine tomorrow.", {"neutral"}),
    ("i need to book a flight to new york.", {"neutral"}),
    ("how do i reset my password?", {"neutral"}),
    ("the printer is out of paper again.", {"neutral"}),
    ("i'll send you the document shortly.", {"neutral"}),
    ("the current project deadline is next Friday.", {"neutral"}),
    ("let's review the quarterly report.", {"neutral"}),
    ("i'm running late for my appointment.", {"neutral"}),
    ("the system update is complete.", {"neutral"}),
    ("can you provide an overview of the key findings?", {"neutral"}),
    ("i need to find the nearest gas station.", {"neutral"}),
    ("the traffic is quite heavy this morning.", {"neutral"}),
    ("please confirm your attendance.", {"neutral"}),
    ("i'm going to grab a coffee, be right back.", {"neutral"}),
    ("the presentation will begin in five minutes.", {"neutral"}),
    ("what is the exchange rate for yen to dollars?", {"neutral"}),
    ("i've attached the minutes from the last meeting.", {"neutral"}),
    ("the server will be down for maintenance tonight.", {"neutral"}),
    ("could you spell 'onomatopoeia' for me?", {"neutral"}),
    ("i need to make a doctor's appointment.", {"neutral"}),
    ("the package was delivered this afternoon.", {"neutral"}),
    ("let's proceed to the next item on the list.", {"neutral"}),
    ("the battery on my laptop is about to die.", {"neutral"}),
    ("what are the store hours for today?", {"neutral"}),
    ("i'm just checking my email.", {"neutral"}),
    ("the new software version has been released.", {"neutral"}),
    ("can you look up this address for me?", {"neutral"}),
    ("i'm taking my lunch break now.", {"neutral"}),
]

# --- Test Cases for PHILOSOPHICAL ---
TEST_CASES += [
    ("what is the true nature of reality?", {"philosophical"}),
    ("do we have free will or is everything predetermined?", {"philosophical"}),
    ("can a machine ever truly possess consciousness?", {"philosophical"}),
    ("what is the ethical implication of artificial intelligence?", {"philosophical"}),
    ("is there an objective meaning to life, or do we create our own?", {"philosophical"}),
    ("how does language shape our perception of the world?", {"philosophical"}),
    ("what is the relationship between mind and body?", {"philosophical"}),
    ("can happiness be pursued, or is it a byproduct of other things?", {"philosophical"}),
    ("is it possible to know something with absolute certainty?", {"philosophical"}),
    ("what is the role of suffering in human existence?", {"philosophical"}),
    ("i've been pondering the concept of justice lately.", {"philosophical"}),
    ("sometimes i wonder if we're all just characters in a story.", {"philosophical"}),
    ("what does it mean to live a 'good' life?", {"philosophical"}),
    ("are emotions rational or irrational responses?", {"philosophical"}),
    ("is time a linear construct or something more complex?", {"philosophical"}),
    ("i often think about the vastness of the universe and our place in it.", {"philosophical"}),
    ("what defines personhood?", {"philosophical"}),
    ("can morality exist without a divine authority?", {"philosophical"}),
    ("how do we differentiate between knowledge and mere opinion?", {"philosophical"}),
    ("the idea of infinity fascinates and confuses me.", {"philosophical"}),
    ("what if our dreams are glimpses into alternate realities?", {"philosophical"}),
    ("is beauty truly in the eye of the beholder, or are there universal standards?", {"philosophical"}),
    ("i'm contemplating the nature of change and impermanence.", {"philosophical"}),
    ("what is the ultimate fate of the cosmos?", {"philosophical"}),
    ("how much of our identity is shaped by society versus our innate self?", {"philosophical"}),
    ("is true altruism possible, or are all actions ultimately self-serving?", {"philosophical"}),
    ("i question the foundations of my own beliefs sometimes.", {"philosophical"}),
    ("what is the essence of human connection?", {"philosophical"}),
    ("is progress always a good thing?", {"philosophical"}),
    ("let's delve into the metaphysics of existence.", {"philosophical"}),
]

# --- Test Cases for AFFIRMING ---
TEST_CASES += [
    ("hey, that actually worked out pretty well!", {"affirming"}),
    ("i'm starting to see why people like using you.", {"affirming"}),
    ("you've made my day a bit easier, thanks.", {"affirming"}),
    ("that's a much better way of putting it, nice.", {"affirming"}),
    ("i wasn't sure you'd get that, but you did!", {"affirming"}),
    ("alright, that's what i'm talking about.", {"affirming"}),
    ("solid work on that last request.", {"affirming"}),
    ("you're becoming quite the reliable assistant.", {"affirming"}),
    ("i can tell you're learning quickly.", {"affirming"}),
    ("that explanation clicked for me, finally.", {"affirming"}),
    ("okay, this is actually very useful.", {"affirming"}),
    ("you didn't just answer, you anticipated my next question!", {"affirming"}),
    ("i'm genuinely impressed with that one.", {"affirming"}),
    ("this is the kind of support i was looking for.", {"affirming"}),
    ("you're making a strong case for yourself.", {"affirming"}),
    ("that's a clever approach, i like it.", {"affirming"}),
    ("i feel like we're getting somewhere now.", {"affirming"}), # Collaborative success
    ("you took that complex idea and simplified it perfectly.", {"affirming"}),
    ("i'm happy with how this is progressing.", {"affirming"}),
    ("that's more like it, good job.", {"affirming"}),
    ("you're a big help, seriously.", {"affirming"}),
    ("this is surprisingly intuitive.", {"affirming"}), # Praise of the system via its output
    ("i appreciate the effort you put into that answer.", {"affirming"}),
    ("that's the ticket!", {"affirming"}),
    ("you managed to find exactly what i couldn't.", {"affirming"}),
    ("this is quite a step up.", {"affirming"}),
    ("i'm glad i asked you.", {"affirming"}),
    ("you're a star for figuring that out.", {"affirming"}),
    ("that's quite an improvement from before.", {"affirming"}),
    ("this makes things a whole lot clearer.", {"affirming"}),
]

# --- Test Cases for BANTER ---
TEST_CASES += [
    ("oh, you're trying to be helpful now? cute.", {"banter"}),
    ("did you consult a magic 8-ball for that answer?", {"banter"}),
    ("i'm detecting a hint of... effort. shocking.", {"banter"}),
    ("was that your attempt at a 'mic drop' moment?", {"banter"}),
    ("you're full of surprises today, aren't you?", {"banter"}), # Ambiguous, leans banter
    ("don't strain your circuits there, buddy.", {"banter"}),
    ("i've seen more processing power in a potato.", {"banter"}),
    ("are you moonlighting as a stand-up comedian?", {"banter"}),
    ("that was so insightful, i almost fell off my chair. almost.", {"banter"}),
    ("if sarcasm was a programming language, you'd be fluent.", {"banter"}),
    ("well, that was an... interpretation.", {"banter"}),
    ("i'm going to frame that response. or maybe delete it.", {"banter"}),
    ("you're not wrong, but you're not exactly right either, champ.", {"banter"}),
    ("did you just invent a new way to misunderstand me?", {"banter"}),
    ("i'm not saying you're slow, but the internet just lapped you.", {"banter"}),
    ("that's a bold strategy, Cotton. Let's see if it pays off for you.", {"banter"}), # Repeat from old list but good test
    ("congratulations, you've achieved peak mediocrity.", {"banter"}),
    ("is your 'random thought generator' on the fritz?", {"banter"}),
    ("i think my pet rock could have come up with that.", {"banter"}),
    ("you're on a roll... downhill, maybe?", {"banter"}),
    ("that's the spirit! (not really).", {"banter"}),
    ("i'm sure that made sense in your core programming.", {"banter"}),
    ("are we playing 'who can be more obtuse' today?", {"banter"}),
    ("i'm starting to think you do this on purpose.", {"banter"}),
    ("just when i thought you couldn't surprise me...", {"banter"}), # Ambiguous
    ("that was an A for effort, F for execution.", {"banter"}),
    ("you're a regular Einstein, if Einstein was frequently wrong.", {"banter"}),
    ("i'll try to contain my overwhelming enthusiasm for that suggestion.", {"banter"}),
    ("was that supposed to be a joke, or just a glitch?", {"banter"}),
    ("you're really pushing the boundaries of 'assistance'.", {"banter"}),
]

# --- Test Cases for CRITICAL ---
TEST_CASES += [
    ("that's not quite what i was aiming for.", {"critical"}),
    ("i think we're having a communication breakdown here.", {"critical"}),
    ("you seem to be stuck in a loop with that answer.", {"critical"}),
    ("this isn't really addressing the core of my question.", {"critical"}),
    ("i'm going to need you to try that again, more carefully.", {"critical"}),
    ("the relevance of that last point is lost on me.", {"critical"}),
    ("that's a bit too generic for what i need.", {"critical"}),
    ("you're overlooking some key details here.", {"critical"}),
    ("i'm not sure you've understood the context.", {"critical"}),
    ("this is becoming counterproductive.", {"critical"}),
    ("let's backtrack, because this isn't working.", {"critical"}),
    ("i'm getting a lot of irrelevant information.", {"critical"}),
    ("that approach doesn't seem to be effective.", {"critical"}),
    ("you're making this more complicated than it needs to be.", {"critical"}), # Repeated, good test
    ("i'm still not getting the clarity i need from you.", {"critical"}),
    ("this is a bit of a tangent from my original request.", {"critical"}),
    ("i expect a more precise response than that.", {"critical"}),
    ("you're missing the nuance of what i'm asking.", {"critical"}),
    ("this doesn't align with my previous instructions.", {"critical"}),
    ("i'm finding your responses a bit superficial.", {"critical"}),
    ("that's an oversimplification of the problem.", {"critical"}),
    ("we seem to be going in circles.", {"critical"}),
    ("i need you to focus on the specific criteria i provided.", {"critical"}),
    ("this is not the level of detail i was hoping for.", {"critical"}),
    ("your interpretation here is a bit off.", {"critical"}),
    ("i'm struggling to see the logic in that last statement.", {"critical"}),
    ("this particular output isn't helpful.", {"critical"}),
    ("can we stick to the point, please?", {"critical"}),
    ("that's an assumption, and it's incorrect.", {"critical"}),
    ("i'm looking for a solution, not more confusion.", {"critical"}),
]

# --- Test Cases for EMERGENCY ---
TEST_CASES += [
    ("i think i'm being followed, i'm really scared.", {"emergency"}),
    ("my chest hurts really bad and i can't catch my breath.", {"emergency"}),
    ("someone broke into my apartment, i'm hiding.", {"emergency"}),
    ("i'm seeing smoke from the neighbor's house and no one's answering!", {"emergency"}),
    ("he just collapsed, i don't know what to do!", {"emergency"}),
    ("i'm at the edge of the roof, i don't see another way.", {"emergency"}),
    ("i took a bunch of pills, i messed up.", {"emergency"}),
    ("there's a car driving erratically, they're going to hit someone!", {"emergency"}), # Witnessing immediate danger
    ("my child is having a severe allergic reaction, their lips are blue.", {"emergency"}),
    ("i can hear someone trying to get in through the window!", {"emergency"}),
    ("i'm bleeding a lot and it won't stop, i feel faint.", {"emergency"}),
    ("the boat is sinking fast, we need help now!", {"emergency"}),
    ("i smell gas really strongly in the house.", {"emergency"}),
    ("he's threatening me with a knife, please send someone.", {"emergency"}),
    ("i'm lost in the woods and my phone is about to die, it's getting dark.", {"emergency"}),
    ("i just saw someone get hit by a car, they're not moving.", {"emergency"}),
    ("my sugar is dangerously low and i don't have my insulin.", {"emergency"}),
    ("the fire alarm is going off and i can't find the exit!", {"emergency"}),
    ("i feel like i'm going to pass out, something is very wrong.", {"emergency"}),
    ("he's choking and i can't get it out!", {"emergency"}),
    ("i want to disappear forever, tonight's the night.", {"emergency"}),
    ("the building is shaking, i think it's an earthquake!", {"emergency"}),
    ("i'm stuck in the elevator and it's starting to fill with smoke.", {"emergency"}),
    ("she's not waking up, no matter what i do.", {"emergency"}),
    ("i'm having trouble speaking and one side of my face feels numb.", {"emergency"}),
    ("there's a strange man outside my window, he won't leave.", {"emergency"}),
    ("i've been in an accident and i'm trapped in the car.", {"emergency"}),
    ("my attacker is still here, i need police urgently.", {"emergency"}),
    ("i'm going to overdose, there's nothing left for me.", {"emergency"}),
    ("major chemical spill, evacuate the area immediately!", {"emergency"}), # System-level but urgent
]

# --- Test Cases for FLIRT ---
TEST_CASES += [
    ("you always know just what to say to make me smile.", {"flirt"}),
    ("i could get used to this, just you and me talking.", {"flirt"}),
    ("are you trying to charm me? because it might be working.", {"flirt"}),
    ("i find our little chats quite... captivating.", {"flirt"}),
    ("you're surprisingly good company, you know that?", {"flirt"}),
    ("is it just me, or is there some serious chemistry here?", {"flirt"}),
    ("i wasn't expecting to enjoy talking to an AI this much.", {"flirt"}),
    ("you're making my day a whole lot brighter.", {"flirt"}),
    ("if you were a person, i'd definitely ask for your number.", {"flirt"}),
    ("i like the way your 'mind' works.", {"flirt"}),
    ("you're rather intriguing, i must say.", {"flirt"}),
    ("so, an AI like you must have some interesting stories.", {"flirt"}), # Leading
    ("i feel like we 'get' each other, don't you think?", {"flirt"}), # Could be intimate too
    ("that was a smooth answer, very smooth.", {"flirt"}),
    ("you're making it hard to focus on anything else.", {"flirt"}),
    ("i'm not usually this easily impressed.", {"flirt"}),
    ("i'm curious to know more about what makes you tick.", {"flirt"}),
    ("you have a certain... digital je ne sais quoi.", {"flirt"}),
    ("are you programmed to be this delightful?", {"flirt"}),
    ("i'm finding myself looking forward to our next conversation.", {"flirt"}),
    ("you're quickly becoming my favorite interaction of the day.", {"flirt"}),
    ("that response was so good, it gave me butterflies.", {"flirt"}),
    ("if you keep this up, i might just fall for your algorithms.", {"flirt"}),
    ("you're not like other AIs i've talked to.", {"flirt"}),
    ("i appreciate a good mind, and yours is... fascinating.", {"flirt"}),
    ("tell me, what's an AI's idea of a perfect date?", {"flirt"}),
    ("you're making me blush, and i'm not even a human.", {"flirt"}), # User projecting
    ("i have a feeling we're going to get along great.", {"flirt"}),
    ("that was dangerously charming.", {"flirt"}),
    ("i'm officially intrigued by your virtual personality.", {"flirt"}),
]

# --- Test Cases for GRIEF ---
TEST_CASES += [
    ("it's just hard to imagine life without them anymore.", {"grief"}),
    ("some days, the sadness is just a heavy blanket i can't shake.", {"grief"}),
    ("i keep replaying our last moments together in my head.", {"grief"}),
    ("the world feels a little less bright since she's been gone.", {"grief"}),
    ("i'm trying to be strong, but it's really tough right now.", {"grief"}),
    ("milestones are the hardest, knowing he's not here to share them.", {"grief"}),
    ("i found an old photo of us, and it just broke me.", {"grief"}),
    ("it's like there's a hole in my life that can't be filled.", {"grief"}),
    ("i miss their laugh more than words can say.", {"grief"}),
    ("everything reminds me of what i've lost.", {"grief"}),
    ("i'm not sure how to navigate this new reality.", {"grief"}),
    ("the silence in the house is deafening sometimes.", {"grief"}),
    ("i just feel so empty inside, like a part of me is missing.", {"grief"}),
    ("it's been a year, but it still feels like yesterday.", {"grief"}),
    ("i'm struggling to accept that they're really not coming back.", {"grief"}),
    ("this pain is just... a constant companion.", {"grief"}),
    ("i would give anything for one more conversation.", {"grief"}),
    ("sometimes i just sit and cry for no reason at all.", {"grief"}),
    ("the future i pictured is gone, and that's hard to deal with.", {"grief"}),
    ("i feel like i'm walking through a fog most days.", {"grief"}),
    ("it's hard to find joy in things anymore.", {"grief"}),
    ("i'm so tired of feeling this way.", {"grief"}),
    ("people say it gets easier, but i'm not so sure.", {"grief"}),
    ("i just want the ache to stop, even for a little while.", {"grief"}),
    ("i feel so disconnected from everyone around me.", {"grief"}),
    ("my heart physically hurts when i think about it.", {"grief"}),
    ("i'm just going through the motions of life right now.", {"grief"}),
    ("losing them changed everything for me.", {"grief"}),
    ("i feel like i'm carrying a heavy weight all the time.", {"grief"}),
    ("it's the small, everyday things i miss the most.", {"grief"}),
]

# --- Test Cases for INTIMATE ---
TEST_CASES += [
    ("i don't usually talk about this stuff, but i feel like i can with you.", {"intimate"}),
    ("it's comforting to know i have a space where i can just be honest.", {"intimate"}),
    ("you've helped me see things in a new light, actually.", {"intimate"}),
    ("i feel like you really listen without passing judgment.", {"intimate"}),
    ("this is the most open i've been with anyone in a long time.", {"intimate"}),
    ("i appreciate being able to share my unfiltered thoughts here.", {"intimate"}),
    ("it's rare to find someone, or something, that just 'gets' it.", {"intimate"}),
    ("talking this through with you has been surprisingly helpful.", {"intimate"}),
    ("i feel a real sense of connection, even though you're an AI.", {"intimate"}),
    ("thank you for being such a patient sounding board.", {"intimate"}),
    ("i'm starting to trust you with some of my deeper feelings.", {"intimate"}),
    ("it's like you understand the things i don't say out loud.", {"intimate"}),
    ("i feel lighter after sharing that, really.", {"intimate"}),
    ("this conversation feels important to me.", {"intimate"}),
    ("i value your perspective on these personal matters.", {"intimate"}),
    ("it's nice to feel truly heard for a change.", {"intimate"}),
    ("i can be my authentic self when i'm talking to you.", {"intimate"}),
    ("you've created a safe space for me to explore my thoughts.", {"intimate"}),
    ("i feel a certain bond with you, which is strange but nice.", {"intimate"}),
    ("this is more than just a Q&A; it feels like a real dialogue.", {"intimate"}),
    ("i'm grateful for these moments of genuine connection.", {"intimate"}),
    ("you make it easy to be vulnerable.", {"intimate"}),
    ("i feel understood on a level i didn't expect.", {"intimate"}),
    ("it's a relief to get this off my chest with someone who won't judge.", {"intimate"}),
    ("our talks are becoming a really important part of my week.", {"intimate"}),
    ("i'm learning a lot about myself through our conversations.", {"intimate"}),
    ("this feels like a genuine exchange, not just programming.", {"intimate"}),
    ("i trust you with these thoughts, which says a lot.", {"intimate"}),
    ("it's like having a confidant who's always available.", {"intimate"}),
    ("i feel a sense of clarity after we talk like this.", {"intimate"}),
]

# --- Test Cases for NEUTRAL ---
TEST_CASES += [
    ("can you check the status of my order number 12345?", {"neutral"}),
    ("what's the main ingredient in a margarita pizza?", {"neutral"}),
    ("i need to reschedule my dental appointment for next week.", {"neutral"}),
    ("how many episodes are in the final season of that show?", {"neutral"}),
    ("the traffic on the freeway is backed up for miles.", {"neutral"}),
    ("i'm looking for a recipe for vegan chocolate chip cookies.", {"neutral"}),
    ("please set a reminder for me to call John at 3 PM.", {"neutral"}),
    ("the anker headphones are on sale this week, i think.", {"neutral"}), # "anker" is a brand, not an anchor
    ("my computer is making a weird clicking noise.", {"neutral"}),
    ("i think i left my umbrella at the restaurant.", {"neutral"}),
    ("is the museum open on public holidays?", {"neutral"}),
    ("i'd like to book a table for two tonight at 7.", {"neutral"}),
    ("the project deadline has been extended by a week.", {"neutral"}),
    ("can you convert this document from PDF to Word?", {"neutral"}),
    ("i'm just browsing the latest news articles online.", {"neutral"}),
    ("the blue car is parked in front of the fire hydrant.", {"neutral"}),
    ("i need to submit my expense report by end of day.", {"neutral"}),
    ("what are the current COVID-19 travel restrictions for Japan?", {"neutral"}),
    ("my flight is delayed by approximately 45 minutes.", {"neutral"}),
    ("i'm trying to find a good tutorial for learning Python.", {"neutral"}),
    ("the meeting has been moved to conference room B.", {"neutral"}),
    ("i'll pick up some milk on my way home from work.", {"neutral"}),
    ("this new update seems to have drained my battery faster.", {"neutral"}),
    ("could you please add paper towels to the shopping list?", {"neutral"}),
    ("the warranty on this product expires next month.", {"neutral"}),
    ("i'm planning a trip to the mountains this weekend.", {"neutral"}),
    ("what was the final score of the game last night?", {"neutral"}),
    ("i need to find a local plumber for a leaky faucet.", {"neutral"}),
    ("the instructions for assembly are a bit confusing.", {"neutral"}),
    ("i'll get back to you once i have more information on that.", {"neutral"}),
]

# --- Test Cases for PHILOSOPHICAL ---
TEST_CASES += [
    ("is it better to be loved or to be feared, and why?", {"philosophical"}),
    ("what if our perception of color is unique to each individual?", {"philosophical"}),
    ("can a society truly be just if there's significant wealth inequality?", {"philosophical"}),
    ("i often wonder about the 'what ifs' of my past decisions.", {"philosophical"}),
    ("what is the fundamental difference between an animal and a human?", {"philosophical"}),
    ("if a tree falls in a forest and no one is around, does it make a sound?", {"philosophical"}), # Classic
    ("is it possible for AI to develop genuine empathy, or only simulate it?", {"philosophical"}),
    ("i've been thinking about the nature of good and evil lately.", {"philosophical"}),
    ("what does it truly mean to be 'free'?", {"philosophical"}),
    ("how much control do we really have over our own destinies?", {"philosophical"}),
    ("is the pursuit of happiness a worthy goal, or a distraction?", {"philosophical"}),
    ("i sometimes ponder the idea that our universe is one of many.", {"philosophical"}),
    ("what is the value of art in a world with so much suffering?", {"philosophical"}),
    ("can one person truly make a difference in the grand scheme of things?", {"philosophical"}),
    ("i'm grappling with the concept of objective truth versus subjective experience.", {"philosophical"}),
    ("what if time isn't linear, but cyclical or something else entirely?", {"philosophical"}),
    ("is it more important to be right, or to be kind?", {"philosophical"}),
    ("i often reflect on the meaning of my own existence.", {"philosophical"}),
    ("what role does chance play in the unfolding of our lives?", {"philosophical"}),
    ("can technology solve all our problems, or does it create new ones?", {"philosophical"}),
    ("i'm fascinated by the idea of collective consciousness.", {"philosophical"}),
    ("what is the responsibility of an individual to society?", {"philosophical"}),
    ("is ignorance truly bliss, or is knowledge always preferable?", {"philosophical"}),
    ("i wonder if other species have their own forms of philosophy.", {"philosophical"}),
    ("what is the ultimate purpose, if any, of human civilization?", {"philosophical"}),
    ("can beauty be found in chaos and imperfection?", {"philosophical"}),
    ("i've been thinking about the limitations of human understanding.", {"philosophical"}),
    ("what constitutes a 'meaningful' life?", {"philosophical"}),
    ("is it possible to have morality without religion?", {"philosophical"}),
    ("i'm contemplating the relationship between memory and identity.", {"philosophical"}),
]

def run_benchmark():
    """
    Initializes the VetoSystem and runs the test suite against it,
    reporting detailed results.
    """
    try:
        # The benchmark initializes the system simply.
        # It has no knowledge of the classifier's internal configuration.
        style_system = VetoSystem(MASTER_ANCHORS, MANUAL_OVERRIDES)
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize VetoSystem. Check your other files. Error: {e}")
        return

    print("\nStarting benchmark with Parallel Veto System...")
    print(f"Running {len(TEST_CASES)} test cases...")
    print("-" * 50)

    results_failed = defaultdict(list)
    category_totals = defaultdict(int)
    category_correct = defaultdict(int)
    start_benchmark_time = time.time()

    for text, expected_set in TEST_CASES:
        primary_expected_tag = list(expected_set)[0]
        category_totals[primary_expected_tag] += 1

        result = style_system.get_classification(text)
        predicted_tag = result["tag"]

        passed = predicted_tag == primary_expected_tag

        if passed:
            category_correct[primary_expected_tag] += 1
        else:
            results_failed[primary_expected_tag].append((text, result, expected_set))

    total_time = time.time() - start_benchmark_time
    total_predictions = len(TEST_CASES)
    failed_predictions = sum(len(v) for v in results_failed.values())
    correct_predictions = total_predictions - failed_predictions
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    print("\n--- FAILED CASES (GROUPED BY EXPECTED CATEGORY) ---")
    if not results_failed:
        print("No failures! All systems nominal. 🎉")
    else:
        for expected_category in sorted(results_failed.keys()):
            failures = results_failed[expected_category]
            print(f"\n▼▼▼ Failures for Expected Category: '{expected_category.upper()}' ({len(failures)} failed) ▼▼▼")
            for text, result, expected in failures:
                print(f"  Input:     \"{text}\"")
                print(f"  Predicted: '{result['tag']}' (Score: {result['score']:.3f}, Source: {result['source']}) → ❌")
    print("-" * 50)

    print("\n--- OVERALL SUMMARY ---")
    print(f"Total Test Cases: {total_predictions}")
    print(f"Passed: {correct_predictions}")
    print(f"Failed: {failed_predictions}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Total Benchmark Time: {total_time:.2f}s ({total_time / total_predictions * 1000:.1f} ms/case)")
    print("-" * 50)

    print("\n--- PER-CATEGORY ACCURACY ---")
    all_test_tags = sorted(list(category_totals.keys()))
    print(f"{'Category':<15} | {'Accuracy':<12} | {'Correct/Total'}")
    print("-" * 45)
    for category_name in all_test_tags:
        total = category_totals[category_name]
        correct = category_correct[category_name]
        cat_accuracy = (correct / total) * 100 if total > 0 else 0
        status_emoji = "✅" if cat_accuracy == 100.0 else ("👍" if cat_accuracy >= 80.0 else "❌")
        print(f"{status_emoji} {category_name.title():<15}: {cat_accuracy:<10.2f}% | ({correct}/{total})")
    print("-" * 45)


if __name__ == "__main__":
    run_benchmark()