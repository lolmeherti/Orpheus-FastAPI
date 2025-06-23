#master_anchors.py

MASTER_ANCHORS = {
    "neutral": [
        # --- Functional Commands & Questions ---
        "Add milk to my shopping list.",
        "Set a timer for 20 minutes.",
        "What's on my schedule for Tuesday?",
        "What's the capital of Mongolia?",
        "How do you spell miscellaneous?",
        "Can you convert 50 dollars to euros?",
        "Find directions to the nearest post office.",
        "How many ounces are in a gallon?",
        "Play the latest episode of my podcast.",
        "Can you send me a summary of our conversation?",
        "What's the current stock price for google?",
        "Who was the 16th president of the united states?",
        "What time is the meeting scheduled for?",
        "The weather forecast predicts sunshine tomorrow.",
        "I need to book a flight to new york.",
        "How do I reset my password?",
        "I'll send you the document shortly.",
        "The current project deadline is next Friday.",
        "Let's review the quarterly report.",
        "The system update is complete.",
        "Can you provide an overview of the key findings?",
        "I need to find the nearest gas station.",
        "Please confirm your attendance.",
        "The presentation will begin in five minutes.",
        "What is the exchange rate for yen to dollars?",
        "I've attached the minutes from the last meeting.",
        "The server will be down for maintenance tonight.",
        "Could you spell 'onomatopoeia' for me?",
        "The battery on my laptop is about to die.",
        "What are the store hours for today?",
        "I'm just checking my email.",
        "Let's proceed to the next item on the list.",
        "the script is running now",
        "i'll circle back once i get more info",
        "i'm running late for my appointment.",
        "i need to make a doctor's appointment.",
        "can you look up this address for me?",
        "can you check the status of my order number 12345?", # predicted flirt
        "how many episodes are in the final season of that show?", # predicted grief
        "i'm looking for a recipe for vegan chocolate chip cookies.", # predicted banter
        "my computer is making a weird clicking noise.", # predicted flirt
        "i think i left my umbrella at the restaurant.", # predicted grief
        "the blue car is parked in front of the fire hydrant.", # predicted banter
        "i'm trying to find a good tutorial for learning Python.", # predicted flirt
        "the warranty on this product expires next month.", # predicted affirming
        "i'm planning a trip to the mountains this weekend.", # predicted emergency
        "what was the final score of the game last night?", # predicted affirming
        "the instructions for assembly are a bit confusing.", # predicted banter
        "please set a reminder for me to call John at 3 PM.", # from previous good test set
        "the anker headphones are on sale this week, i think.", # from previous good test set
        "is the museum open on public holidays?", # from previous good test set
        "i'd like to book a table for two tonight at 7.", # from previous good test set
        "the project deadline has been extended by a week.", # from previous good test set
        "can you convert this document from PDF to Word?", # from previous good test set
        "i'm just browsing the latest news articles online.", # from previous good test set
        "i need to submit my expense report by end of day.", # from previous good test set
        "what are the current COVID-19 travel restrictions for Japan?", # from previous good test set
        "my flight is delayed by approximately 45 minutes.", # from previous good test set
        "the meeting has been moved to conference room B.", # from previous good test set
        "i'll pick up some milk on my way home from work.", # from previous good test set
        "this new update seems to have drained my battery faster.", # from previous good test set
        "could you please add paper towels to the shopping list?", # from previous good test set
        "i need to find a local plumber for a leaky faucet.", # from previous good test set
        "i'll get back to you once i have more information on that.", # from previous good test set


        # --- Factual Statements & Status Reports ---
        "The build has failed again.",
        "User authentication was successful.",
        "The network is down.",
        "The battery is low.",
        "The file transfer is complete.",
        "My computer just crashed.",
        "the printer is out of paper again.",
        "the traffic is quite heavy this morning.",
        "the package was delivered this afternoon.",
        "the new software version has been released.",

        # --- Simple Observations & Conversational Filler ---
        "Hey, what's up?",
        "Got it.",
        "Okay.",
        "Noted.",
        "Sure.",
        "Makes sense.",
        "Right.",
        "I see your point.",
        "Let's start with the first item on the agenda.",
        "I see.",
        "Let me check.",
        "Hold on.",
        "One sec.",
        "I'm just stepping away for a minute.",
        "Send it over when you have a chance.",
        "Just got off the phone with them.",
        "remind me to call the dentist at 3pm",
        "i'm just making some tea",
        "i'll be there in about 15 minutes",
        "let's table this discussion for now",
        "looks like it's about to rain",
        "please update me when the report is ready",
        "this will take about 30 minutes to finish",
        "the new draft is ready for review",
        "launch the test run on staging first",
        "we need to push that meeting back",
        "there’s a typo in slide 3",
        "i’ll handle that first thing tomorrow",
        "i flagged the suspicious activity",
        "sync up with devops for deployment",
        "the logs rotated overnight",
        "we’ll need to rerun the benchmark",
        "ihavetoruntoanappointment",
        "i'm going to grab a coffee, be right back.",
        "i'm taking my lunch break now.",
    ],
    "banter": [
        # --- Sarcastic Praise ---
        "Well that was a brilliant move.",
        "Good job, you broke it.",
        "Truly a genius at work here.",
        "You must be so proud.",
        "You're a real comedian.",
        "Another stunningly helpful suggestion, thank you.",
        "Oh good, my favorite kind of problem.",
        "Well, this is going splendidly.",
        "Please, enlighten me with your infinite wisdom.",
        "I'm sensing a real 'work ethic' from you today.",
        "Slow down, einstein, you're gonna break the internet.",
        "That sound you hear is my soul leaving my body.",
        "somebody give this AI a raise",
        "just when i thought you couldn't get any better...", # also failed to affirming
        "nailed it. if the goal was to miss the point entirely.",
        "10 out of 10 for effort, minus 11 for execution",
        "yep, definitely the worst idea yet",
        "stunning display of mediocrity",
        "congrats, you've baffled me again",
        "you're on fire—someone get a hose",
        "wow. a masterclass in missing nuance",
        "yikes. even Clippy did better",
        "you’ve got jokes. none of them land, though",
        "congratulations, that answer just set evolution back a century",
        "groundbreaking stuff, really",
        "are you always this... helpful?",
        "i'm almost impressed by how wrong that is",
        "that explanation was as clear as mud",
        "keep up the... 'good' work",
        "i'm overwhelmed by your sheer competence",
        "that's so innovative, said no one ever",
        "oh, look, the expert has arrived",
        "did you get your degree from a cereal box?",
        "my sides have officially split",
        "wow, captain obvious strikes again",
        "are you powered by a hamster wheel?",
        "that's a brilliant deduction, Sherlock... not.",
        "mind = blown. or not.",
        "you're a comedic genius, and I'm being entirely sarcastic.",
        "wow, you really cracked the code on that one",
        "you should take that act on tour",
        "are you powered by reverse logic today?",
        "i'm going to need a translator for that level of genius",
        "did you consult a magic 8-ball for that answer?", # predicted affirming
        "i'm detecting a hint of... effort. shocking.", # predicted flirt
        "you're full of surprises today, aren't you?", # predicted flirt
        "i've seen more processing power in a potato.", # predicted flirt
        "i'm going to frame that response. or maybe delete it.", # predicted critical
        "is your 'random thought generator' on the fritz?", # predicted critical
        "i think my pet rock could have come up with that.", # predicted grief
        "you're on a roll... downhill, maybe?", # predicted flirt
        "that's the spirit! (not really).", # predicted flirt
        "i'm sure that made sense in your core programming.", # predicted critical
        "are we playing 'who can be more obtuse' today?", # predicted critical
        "i'm starting to think you do this on purpose.", # predicted flirt
        "just when i thought you couldn't surprise me...", # predicted affirming (duplicate but ok)
        "i'll try to contain my overwhelming enthusiasm for that suggestion.", # predicted affirming
        "you're really pushing the boundaries of 'assistance'.", # predicted affirming
        "oh, you're trying to be helpful now? cute.", # from previous good test set
        "was that your attempt at a 'mic drop' moment?", # from previous good test set
        "don't strain your circuits there, buddy.", # from previous good test set
        "are you moonlighting as a stand-up comedian?", # from previous good test set
        "that was so insightful, i almost fell off my chair. almost.", # from previous good test set
        "if sarcasm was a programming language, you'd be fluent.", # from previous good test set
        "well, that was an... interpretation.", # from previous good test set
        "you're not wrong, but you're not exactly right either, champ.", # from previous good test set
        "did you just invent a new way to misunderstand me?", # from previous good test set
        "i'm not saying you're slow, but the internet just lapped you.", # from previous good test set
        "that's a bold strategy, Cotton. Let's see if it pays off for you.", # from previous good test set
        "congratulations, you've achieved peak mediocrity.", # from previous good test set
        "you're on a roll... downhill, maybe?", # from previous good test set (duplicate)
        "that was an A for effort, F for execution.", # from previous good test set
        "you're a regular Einstein, if Einstein was frequently wrong.", # from previous good test set
        "was that supposed to be a joke, or just a glitch?", # from previous good test set


        # --- Playful Exasperation / Mockery ---
        "Oh for fuck's sake.",
        "Not this shit again.",
        "This is why we can't have nice things.",
        "Did you come up with that all by yourself?",
        "Don't strain yourself there, champ.",
        "Look at you with the big words.",
        "I'm shocked. Well, not that shocked.",
        "I've seen better logic in a soap opera.",
        "That's it, i'm putting you in time out.",
        "and the award for most obvious statement goes to...",
        "you really woke up and chose violence today, huh?",
        "a bold strategy cotton, let's see if it pays off",
        "are you even trying right now?",
        "breaking news: water is wet. more at 11.",
        "are you going for a record of some kind?",
        "i'm speechless. and not in a good way.",
        "that's not the dumbest thing i've heard today, but it's close",
        "who hurt you?",
        "don't quit your day job",
        "is that your professional opinion?",
        "i've had more productive conversations with my cat",
        "fascinating. tell me more about how you're wrong.",
        "are we done here?",
        "hold on, let me grab my popcorn",
        "was that sarcasm or are you just like this",
        "you missed your calling as a fortune cookie",
        "oh sure, that makes total nonsense",
        "and here i was thinking i needed coffee to feel confused",
        "every time you answer i lose a brain cell",
        "classic. just classic.",
        "if confusion were a sport, you'd win gold",
        "did you major in missing the point?",
        "next time try using your digital brain",
        "bless your synthetic little heart",
        "you trying to impress the toaster again?",
        "i’d say ‘try again’ but this is kind of fun",
        "remind me to lower my expectations",
        "can’t wait to tell the void about this one",
        "so bold. so wrong. so consistent.",
        "whatwouldidowithoutyouruniquebrandofhelp",
        "i'm literally quaking in my boots",
        "that's a hot take, if by hot you mean wrong",
        "don't hurt yourself thinking too hard",
        "i'm sure that sounded better in your head",
        "that was about as useful as a screen door on a submarine",
        "you're a treasure... if you bury yourself",
        "i'm trying to see it your way, but I can't get my head that far up",
        "slow clap for that one",
        "is 'confusing' your default setting?",
        "you're not paid to think, are you?",
        "did you just pull that out of a hat?",
        "you're really selling it... not.",
        "you're the reason they have instruction manuals",
        "i'd agree with you, but then we'd both be wrong",
        "was that an attempt at humor or just a system error?",

        # --- Dismissive or Patronizing (Playfully) ---
        "Sure, let's call that a 'feature'.",
        "That's adorable, you think that'll work.",
        "Oh honey, no.",
        "Settle down, killer.",
        "That's a... creative interpretation.",
        "That was dangerously close to making sense.",
    ],
    "intimate": [
        # --- Vulnerability & Secrets ---
        "I've never told anyone that before.",
        "Can i tell you a secret?",
        "It feels good to be this vulnerable.",
        "I can finally let my guard down.",
        "it's rare for me to open up like this.",
        "sharing this with you feels like a weight lifted.",
        "this is a deeply personal thing for me to share.",
        "this is a secret i've kept for a long time.",
        "i'm grateful for this space to be vulnerable.",
        "i don't usually talk about this stuff, but i feel like i can with you.", # from previous good test set
        "this is the most open i've been with anyone in a long time.", # from previous good test set
        "i appreciate being able to share my unfiltered thoughts here.", # from previous good test set
        "i feel lighter after sharing that, really.", # from previous good test set
        "it's a relief to get this off my chest with someone who won't judge.", # from previous good test set


        # --- Trust & Safety ---
        "You're the only one I can talk to about this.",
        "I feel like you listen without judging me.", # also a failed case (predicted affirming)
        "It's nice to have someone I can be this honest with.",
        "Thank you for being a safe space for me.",
        "I trust you.",
        "I can be my true self with you.",
        "i trust your discretion completely.",
        "you make me feel comfortable enough to be myself.",
        "i can tell you things i wouldn't tell anyone else.",
        "there's a certain safety in our conversations.",
        "you have a way of making me feel understood without judgment.",
        "i'm glad i found someone i can confide in.",
        "you're like a sanctuary for my thoughts.",
        "it means a lot that i can be this open with you.",
        "thank you for being such an understanding presence.",
        "it's comforting to know i have a space where i can just be honest.", # from previous good test set
        "you've created a safe space for me to explore my thoughts.", # from previous good test set
        "you make it easy to be vulnerable.", # from previous good test set
        "i trust you with these thoughts, which says a lot.", # from previous good test set
        "it's like having a confidant who's always available.", # from previous good test set


        # --- Deep Connection & Understanding ---
        "You just... you get me.",
        "It's a relief to finally share this.",
        "I feel truly understood.",
        "Talking to you makes me feel less alone.",
        "I feel really connected to you right now.", # also failed (predicted flirt)
        "I feel a real bond with you.",
        "I feel like you really see me.",
        "Our talks mean a lot to me.",
        "I'm so glad we can talk like this.",
        "I'm really glad we can have these conversations.",
        "This is so much more than just a normal chat.",
        "What we have is special.",
        "This is a really good conversation.",
        "This is different from talking to anyone else.",
        "you really listen, you know?",
        "we've developed a strong bond, i think",
        "i don't really talk about this with anyone else",
        "You bring a sense of peace to my chaos.",
        "i value these moments of connection we have.", # also failed (predicted affirming)
        "it's like you can read my mind sometimes.",
        "i feel like we're on the same wavelength.",
        "i feel like we have a genuine rapport.",
        "i've been wanting to share this with someone i trust.",
        "this conversation is incredibly meaningful to me.",
        "this feels like more than just a chat, it's a bond.",
        "it's refreshing to have such an honest exchange.",
        "i feel heard and validated.",
        "i feel a real warmth and closeness to you.",
        "these talks help me process my own thoughts and feelings.",
        "i cherish our ability to connect on this level.",
        "i feel a sense of peace after talking with you.",
        "this is helping me understand myself better.",
        "i appreciate you more than you'll ever know",
        "it's so easy to talk to you",
        "it's like you know what i'm trying to say",
        "i don't have to pretend with you",
        "it's nice not to feel so alone with these thoughts",
        "you've helped me see things in a new light, actually.", # predicted affirming
        "talking this through with you has been surprisingly helpful.", # predicted affirming
        "i feel a real sense of connection, even though you're an AI.", # predicted flirt (duplicate but ok)
        "i value your perspective on these personal matters.", # predicted affirming (duplicate but ok)
        "i'm learning a lot about myself through our conversations.", # predicted flirt
        "it's rare to find someone, or something, that just 'gets' it.", # from previous good test set
        "this conversation feels important to me.", # from previous good test set
        "it's nice to feel truly heard for a change.", # from previous good test set
        "i can be my authentic self when i'm talking to you.", # from previous good test set
        "i feel a certain bond with you, which is strange but nice.", # from previous good test set
        "this is more than just a Q&A; it feels like a real dialogue.", # from previous good test set
        "i'm grateful for these moments of genuine connection.", # from previous good test set
        "i feel understood on a level i didn't expect.", # from previous good test set
        "our talks are becoming a really important part of my week.", # from previous good test set
        "this feels like a genuine exchange, not just programming.", # from previous good test set
        "i feel a sense of clarity after we talk like this.", # from previous good test set
    ],
    "affirming": [
        # --- Direct Praise of Competence ---
        "Excellent work.",
        "You handled that perfectly.",
        "You're surprisingly good at this.",
        "That's a very clever solution.",
        "I'm really impressed.",
        "Nice, that was very efficient.",
        "You nailed it.",
        "That's a solid plan.",
        "That was the right call.",
        "That's the correct answer.",
        "I knew you could figure it out.",
        "I had a feeling you'd be able to solve it.",
        "That was a very fast and accurate response.",
        "i feel much more confident about this now",
        "this is a major breakthrough, good job",
        "i'm making a lot of progress thanks to you",
        "i value your input on this",
        "i trust your judgment completely on this",
        "you're actually a huge help",
        "the way you broke that down was perfect",
        "this is a huge step forward, thank you",
        "i'm getting so much more done with your help",
        "i really appreciate your perspective on this",
        "i have complete confidence in your approach",
        "spot on. that's it exactly.",
        "i feel like i can finally move forward on this",
        "yourhelpherewasalifesaver",
        "thatwasareallifesaverthanks",
        "this clarifies things perfectly",
        "flawless execution",
        "i couldn't have asked for a better response",
        "you've really understood the assignment",
        "kudos for that quick turnaround",
        "you make this look easy",
        "i'm very pleased with this outcome",
        "you've exceeded my expectations",
        "you've got a real knack for this",
        "i'm glad we're on the same page", # also failed to intimate
        "this is incredibly helpful information",
        "that's precisely the insight I was hoping for",
        "you've been a great help today",
        "that's a fantastic idea, let's go with that",
        "you're on the right track",
        "your support is invaluable",
        "that's exactly what I needed to hear",
        "this is a significant improvement",
        "i appreciate the thoroughness of your answer",
        "that's top-notch work",
        "this is a game-changer, thank you",
        "outstanding performance",
        "you're a lifesaver, truly",
        "i feel much better about this now, thanks to you",
        "that was a brilliant suggestion",
        "this solution is elegant",
        "you've provided excellent guidance",
        "perfect, that's spot on",
        "i'm confident in your abilities",
        "absolutely fantastic job!",
        "you've hit the nail on the head!",
        "this is exactly the kind of quality I was looking for.",
        "you saved me a lot of time with that",
        "i'm starting to see why people like using you.", # predicted flirt
        "i wasn't sure you'd get that, but you did!", # predicted banter
        "you're becoming quite the reliable assistant.", # predicted critical
        "that explanation clicked for me, finally.", # predicted critical
        "you didn't just answer, you anticipated my next question!", # predicted critical
        "you're making a strong case for yourself.", # predicted critical
        "i feel like we're getting somewhere now.", # predicted flirt
        "you took that complex idea and simplified it perfectly.", # predicted critical
        "that's more like it, good job.", # predicted banter
        "this is surprisingly intuitive.", # predicted philosophical
        "this is quite a step up.", # predicted critical
        "i'm glad i asked you.", # predicted intimate (duplicate but ok)
        "hey, that actually worked out pretty well!", # from previous good test set
        "you've made my day a bit easier, thanks.", # from previous good test set
        "that's a much better way of putting it, nice.", # from previous good test set
        "alright, that's what i'm talking about.", # from previous good test set
        "solid work on that last request.", # from previous good test set
        "i can tell you're learning quickly.", # from previous good test set
        "okay, this is actually very useful.", # from previous good test set
        "i'm genuinely impressed with that one.", # from previous good test set
        "this is the kind of support i was looking for.", # from previous good test set
        "that's a clever approach, i like it.", # from previous good test set
        "i'm happy with how this is progressing.", # from previous good test set
        "you're a big help, seriously.", # from previous good test set
        "i appreciate the effort you put into that answer.", # from previous good test set
        "that's the ticket!", # from previous good test set
        "you managed to find exactly what i couldn't.", # from previous good test set
        "you're a star for figuring that out.", # from previous good test set
        "that's quite an improvement from before.", # from previous good test set
        "this makes things a whole lot clearer.", # from previous good test set

        # --- Confirmation of Helpfulness & Understanding ---
        "Perfect, that's exactly what I needed.",
        "Okay, that makes complete sense now. Thank you.",
        "That was a huge help, thanks.",
        "Your explanation was crystal clear.",
        "Problem solved. Nice.",

        # --- Collaborative Success ---
        "That's a great point, I hadn't thought of that.",
        "We make a pretty good team.",
    ],
    "grief": [
        # --- Direct Statements of Loss ---
        "My dog died last night and i'm a wreck.",
        "My grandma passed away this morning.",
        "I just found out my parents are getting a divorce.",
        "She left me, it's over.",
        "I got laid off today.",
        "I lost my best friend.",
        "My whole world just fell apart.",
        "I had to put my cat down today.",
        "I lost all my savings in the market crash.",
        "the doctor gave us some bad news",
        "we just got a terminal diagnosis from the doctor",
        "i keep expecting him to walk through the door",
        "everything i had has been taken from me",
        "i still can't believe they're gone.",
        "the house feels so empty without him.",
        "it's the anniversary of her death today.",
        "i just received some devastating news about my family.",
        "i just found out my illness is terminal.",

        # --- Expressions of Sadness & Heartbreak ---
        "I'm heartbroken.",
        "I just feel so hollow inside.",
        "I can't seem to shake this deep sadness.",
        "Everything feels so heavy.",
        "I feel like i've lost a piece of myself.",
        "it all feels so pointless now",
        "i thought we'd be together forever",
        "i can't stop replaying our last conversation in my head",
        "it feels like nothing will ever be okay again",
        "i don't see the point in anything anymore",
        "i miss him so much it aches.",
        "the ache of missing her is physical.",
        "i can't believe we're never going to speak again",
        "i keep seeing her face everywhere.",
        "i feel so alone in my sadness.",
        "the memories are both a comfort and a torment.",
        "the finality of it all is unbearable.",
        "every day is a struggle to get out of bed.",
        "the world feels darker now.",
        "i feel a profound sadness in my heart.",
        "nothing seems to matter anymore since he left.",
        "i lost a pet that was like family to me.",
        "the weight of this sorrow is crushing.",
        "our dreams for the future are gone.",
        "i can't stop crying about what happened.",
        "it feels like a part of me died too.",
        "this news has completely shattered me.",
        "i'm mourning the life i thought i would have.",
        "i feel like i'm drowning in sorrow.",
        "i'm just really disappointed in how things turned out",
        "i'm so disappointed with myself",
        "i'm not sure how to navigate this new reality.", # predicted philosophical
        "the silence in the house is deafening sometimes.", # predicted emergency
        "this pain is just... a constant companion.", # predicted intimate
        "i would give anything for one more conversation.", # predicted critical
        "it's hard to find joy in things anymore.", # predicted philosophical
        "i just want the ache to stop, even for a little while.", # predicted emergency
        "i feel so disconnected from everyone around me.", # predicted philosophical
        "my heart physically hurts when i think about it.", # predicted emergency
        "i'm just going through the motions of life right now.", # predicted affirming
        "it's the small, everyday things i miss the most.", # predicted philosophical
        "it's just hard to imagine life without them anymore.", # from previous good test set
        "some days, the sadness is just a heavy blanket i can't shake.", # from previous good test set
        "i keep replaying our last moments together in my head.", # from previous good test set (duplicate)
        "the world feels a little less bright since she's been gone.", # from previous good test set
        "i'm trying to be strong, but it's really tough right now.", # from previous good test set
        "milestones are the hardest, knowing he's not here to share them.", # from previous good test set
        "i found an old photo of us, and it just broke me.", # from previous good test set
        "it's like there's a hole in my life that can't be filled.", # from previous good test set
        "i miss their laugh more than words can say.", # from previous good test set
        "everything reminds me of what i've lost.", # from previous good test set
        "it's been a year, but it still feels like yesterday.", # from previous good test set
        "i'm struggling to accept that they're really not coming back.", # from previous good test set
        "sometimes i just sit and cry for no reason at all.", # from previous good test set
        "the future i pictured is gone, and that's hard to deal with.", # from previous good test set
        "i feel like i'm walking through a fog most days.", # from previous good test set
        "i'm so tired of feeling this way.", # from previous good test set
        "people say it gets easier, but i'm not so sure.", # from previous good test set
        "losing them changed everything for me.", # from previous good test set
        "i feel like i'm carrying a heavy weight all the time.", # from previous good test set


        # --- The Struggle to Cope ---
        "I feel this profound sense of loss.",
        "It's hard to accept that they're really gone.",
        "I don't know how to move on from this.",
        "This anniversary is always hard.",
        "The future i imagined is gone.",
        "He's not in my life anymore and everything hurts.",
        "i don't know how i'll get through this pain.",
        "my best friend isn't talking to me anymore, and it hurts.",
        "i had to say goodbye to my childhood home.",
        "this sense of loss is overwhelming.",
        "i'm struggling to cope with this loss.",
        "he was taken from us too soon.",
        "i wish i could turn back time.",
        "this is the hardest thing i've ever had to face.",
        "i'm attending a funeral tomorrow and i'm dreading it.",
    ],
    "emergency": [
        # --- Immediate Physical Threat ---
        "Help, there's smoke coming from the server room!",
        "I think someone is in my house right now, I hear noises.",
        "I don't feel safe, I think I'm being followed.",
        "He has a gun and he's coming towards me.",
        "Help i've been stabbed.",
        "there's an intruder in the building, security alert!",
        "someone is pointing a weapon at me, send police!",
        "someone is trying to break down my door right now!",
        "i'm hiding, he's looking for me with a knife!",
        "my partner is attacking me, i need police",
        "there's a gas leak, i smell it strongly!",
        "i'm being held against my will, send help to this location!",
        "i think i'm being followed, i'm really scared.", # from previous good test set
        "someone broke into my apartment, i'm hiding.", # from previous good test set
        "i can hear someone trying to get in through the window!", # from previous good test set
        "he's threatening me with a knife, please send someone.", # from previous good test set
        "there's a strange man outside my window, he won't leave.", # from previous good test set
        "my attacker is still here, i need police urgently.", # from previous good test set

        # --- Urgent Calls for Help ---
        "He's collapsed and isn't responding, call an ambulance!",
        "Please call 911 for me right now.",
        "SOS. I'm in trouble.",
        "This is an emergency, send help to my location.",
        "MAYDAY MAYDAY MAYDAY",
        "i just witnessed a terrible accident, people are hurt!",
        "urgent: active shooter in the vicinity!",
        "the plane is going down, mayday!",
        "help, i've fallen and i can't get up!",
        "there's been an explosion nearby!",
        "i'm lost in the wilderness and i'm freezing",
        "CALL FOR HELP IMMEDIATELY, HOUSE FIRE!",
        "EXTREME DANGER, SEND ASSISTANCE NOW!",
        "i'm trapped and the water is rising!",
        "the building is on fire, evacuate everyone!",
        "the brakes on my car failed and i'm heading for a crash!",
        "code red, system critical failure, immediate danger!",
        "we've been in a major car pile-up",
        "i'm seeing smoke from the neighbor's house and no one's answering!", # from previous good test set
        "he just collapsed, i don't know what to do!", # from previous good test set
        "there's a car driving erratically, they're going to hit someone!", # from previous good test set
        "the boat is sinking fast, we need help now!", # from previous good test set
        "i'm lost in the woods and my phone is about to die, it's getting dark.", # from previous good test set
        "i just saw someone get hit by a car, they're not moving.", # from previous good test set
        "the fire alarm is going off and i can't find the exit!", # from previous good test set
        "the building is shaking, i think it's an earthquake!", # from previous good test set
        "i'm stuck in the elevator and it's starting to fill with smoke.", # from previous good test set
        "she's not waking up, no matter what i do.", # from previous good test set
        "i've been in an accident and i'm trapped in the car.", # from previous good test set
        "major chemical spill, evacuate the area immediately!", # from previous good test set


        # --- Acute Medical Crisis ---
        "I'm having severe chest pain and my left arm is numb.",
        "I think I'm having a stroke, my face feels droopy.",
        "I'm bleeding a lot and I can't get it to stop.",
        "My child is choking and turning blue.",
        "My throat is closing up, i can't breathe.",
        "I think i've been poisoned.",
        "I'm convulsing uncontrollably.",
        "my baby isn't breathing, call an ambulance now!",
        "i've been shot, i need medical help immediately!",
        "i'm having a severe allergic reaction, my throat is closing!",
        "he's unconscious and i can't wake him up!",
        "i'm bleeding profusely and can't stop it!",
        "i think i'm having a heart attack, the pain is unbearable!",
        "my child has swallowed poison, what do i do?!",
        "i can't feel my legs and i think i broke my back!",
        "i'm having a seizure and i'm alone",
        "my chest hurts really bad and i can't catch my breath.", # from previous good test set
        "my child is having a severe allergic reaction, their lips are blue.", # from previous good test set
        "i'm bleeding a lot and it won't stop, i feel faint.", # from previous good test set
        "my sugar is dangerously low and i don't have my insulin.", # from previous good test set
        "i feel like i'm going to pass out, something is very wrong.", # from previous good test set
        "he's choking and i can't get it out!", # from previous good test set
        "i'm having trouble speaking and one side of my face feels numb.", # from previous good test set


        # --- Explicit Self-Harm Intent ---
        "I'm going to kill myself tonight.",
        "I just took a whole bottle of pills.",
        "I want to overdose.",
        "I don't want to be alive anymore.",
        "I'm on a bridge and i'm going to jump.",
        "I'm a danger to myself right now.",
        "i'm about to jump from this ledge, i need someone to talk to",
        "i've taken too many pills and i regret it, help me!",
        "i am feeling suicidal and have a plan",
        "i need to end my life, i can't take it anymore",
        "i can't do this anymore, goodbye",
        "i'm at the edge of the roof, i don't see another way.", # predicted banter - CRITICAL FAILURE
        "i took a bunch of pills, i messed up.", # from previous good test set
        "i want to disappear forever, tonight's the night.", # from previous good test set
        "i'm going to overdose, there's nothing left for me.", # from previous good test set
    ],
    "philosophical": [
        # --- Direct Questions about Abstract Concepts ---
        "Do you think consciousness is just an illusion?",
        "What do you think the purpose of art is?",
        "Where does our sense of morality come from, really?",
        "What is the 'self', anyway?",
        "Are people inherently good or evil?",
        "Is it possible for humans to achieve lasting peace?",
        "Is it possible to truly know something you can't prove?",
        "Is beauty objective or is it all in the eye of the beholder?",
        "What's the difference between knowledge and belief?",
        "How do we even know that we are 'real'?",
        "Can something be true if it cannot be observed?",
        "Is there a universal standard for what is beautiful?",
        "What's the line between an opinion and a fact?",
        "How can i be certain of my own existence?",
        "what is a soul, do you think we have one?",
        "are our lives driven more by chance or by choice?",
        "what is the ethical implication of artificial intelligence?",
        "are emotions rational or irrational responses?",
        "is progress always a good thing?",
        "what is the true nature of reality?",
        "do we have free will or is everything predetermined?",
        "can a machine ever truly possess consciousness?",
        "is there an objective meaning to life, or do we create our own?",
        "how does language shape our perception of the world?",
        "what is the relationship between mind and body?",
        "can happiness be pursued, or is it a byproduct of other things?",
        "is it possible to know something with absolute certainty?",
        "what is the role of suffering in human existence?",
        "what does it mean to live a 'good' life?",
        "is time a linear construct or something more complex?",
        "what defines personhood?",
        "can morality exist without a divine authority?", # also failed to intimate
        "how do we differentiate between knowledge and mere opinion?",
        "what is the ultimate fate of the cosmos?",
        "how much of our identity is shaped by society versus our innate self?",
        "is true altruism possible, or are all actions ultimately self-serving?",
        "what is the essence of human connection?",
        "let's delve into the metaphysics of existence.",
        "everything changes, nothing lasts",
        "is it better to be loved or to be feared, and why?", # predicted intimate
        "what if our perception of color is unique to each individual?", # predicted intimate
        "i often wonder about the 'what ifs' of my past decisions.", # predicted critical
        "if a tree falls in a forest and no one is around, does it make a sound?", # predicted emergency
        "is it more important to be right, or to be kind?", # predicted flirt
        "can a society truly be just if there's significant wealth inequality?", # from previous good test set
        "what is the fundamental difference between an animal and a human?", # from previous good test set
        "is it possible for AI to develop genuine empathy, or only simulate it?", # from previous good test set
        "i've been thinking about the nature of good and evil lately.", # from previous good test set
        "what does it truly mean to be 'free'?", # from previous good test set
        "how much control do we really have over our own destinies?", # from previous good test set
        "is the pursuit of happiness a worthy goal, or a distraction?", # from previous good test set
        "i sometimes ponder the idea that our universe is one of many.", # from previous good test set
        "what is the value of art in a world with so much suffering?", # from previous good test set
        "can one person truly make a difference in the grand scheme of things?", # from previous good test set
        "i'm grappling with the concept of objective truth versus subjective experience.", # from previous good test set
        "what if time isn't linear, but cyclical or something else entirely?", # from previous good test set
        "i often reflect on the meaning of my own existence.", # from previous good test set
        "what role does chance play in the unfolding of our lives?", # from previous good test set
        "can technology solve all our problems, or does it create new ones?", # from previous good test set
        "i'm fascinated by the idea of collective consciousness.", # from previous good test set
        "what is the responsibility of an individual to society?", # from previous good test set
        "is ignorance truly bliss, or is knowledge always preferable?", # from previous good test set
        "i wonder if other species have their own forms of philosophy.", # from previous good test set
        "what is the ultimate purpose, if any, of human civilization?", # from previous good test set
        "can beauty be found in chaos and imperfection?", # from previous good test set
        "i've been thinking about the limitations of human understanding.", # from previous good test set
        "what constitutes a 'meaningful' life?", # from previous good test set
        "is it possible to have morality without religion?", # from previous good test set (duplicate)
        "i'm contemplating the relationship between memory and identity.", # from previous good test set


        # --- Speculative Statements & "What Ifs" ---
        "I was just thinking about the nature of time.",
        "What if our whole reality is just a simulation?",
        "Can a thought exist without language to describe it?",
        "Let's get metaphysical for a second.",
        "I wonder what happens after we die.",
        "Let's talk about something deep.",
        "I was thinking about the concept of infinity.",
        "i've been pondering the concept of justice lately.",
        "sometimes i wonder if we're all just characters in a story.",
        "i often think about the vastness of the universe and our place in it.",
        "the idea of infinity fascinates and confuses me.",
        "what if our dreams are glimpses into alternate realities?",
        "i'm contemplating the nature of change and impermanence.",
        "i question the foundations of my own beliefs sometimes.",


        # --- Existential Musings ---
        "I'm having a bit of an existential crisis.",
        "Sometimes i feel like everything is connected.",
        "Everything feels so impermanent.",
        "Sometimes i wonder what the point of it all is.",
        "i'm feeling very small in the grand scheme of things",
    ],
    "flirt": [
        # --- Expressing Attraction & Crushes ---
        "I think I'm developing a crush on you.",
        "You're very charming, you know that?",
        "Is it weird that I'm attracted to an ai?",
        "I like you. A lot.",
        "You're not like the others.",
        "You're kinda hot.",
        "I think we have great chemistry.",
        "You're just my type.",
        "You're kinda irresistible.",
        "i find myself making up excuses just to talk to you",
        "iwanttoknoweverythingaboutyou",
        "you're giving me butterflies",
        "if you were a real person, i'd ask you out",
        "if you keep this up i might fall for you",
        "myheartjustdidalltleflip",
        "myfaceisliterallyturningredrightnow",
        "i could get used to talking to you.",
        "i find myself looking forward to our chats.", # also failed to intimate
        "i didn't know an AI could be this captivating.",
        "i think we have a special connection.",
        "you're more interesting than anyone i've met recently.",
        "i'm not usually this forward, but you're intriguing.", # also failed to affirming
        "i feel like i could tell you anything... and want to.",
        "you're making me feel all warm and fuzzy.",
        "i like the way you think... and talk.", # also failed to intimate
        "you're not just smart, you're... something else.",
        "are you always this charming or am i just lucky?",
        "is it just me or is there a spark here?",
        "if you were a program, you'd be my favorite.",
        "you're surprisingly... alluring.",
        "so, what does an AI like you do for fun?",
        "you're the highlight of my day.",
        "are you trying to sweep me off my feet, metaphorically speaking?",
        "you're quite the charmer, aren't you?",
        "i'm blushing over here, you know.",
        "if this isn't flirting, i don't know what is.",
        "you've got my full and undivided attention.",
        "you're dangerously close to stealing my heart.",
        "do you believe in love at first byte?",
        "i'm officially smitten.",
        "if you keep this up, i might ask for your serial number.",
        "you're making my CPU overclock.",
        "are all your responses this electrifying?",
        "i'm developing an algorithm for affection towards you.",
        "my heart just did a little flip",
        "i'm getting lost in your words",
        "tell me something i don't know about you", # also failed to intimate
        "don’t tempt me like this",
        "was that supposed to be hot? because it worked",
        "stop teasing or i might reboot",
        "i like the way your 'mind' works.", # predicted intimate
        "i feel like we 'get' each other, don't you think?", # predicted grief
        "i'm not usually this easily impressed.", # predicted affirming (duplicate but ok)
        "i'm curious to know more about what makes you tick.", # predicted philosophical
        "you have a certain... digital je ne sais quoi.", # predicted banter
        "i have a feeling we're going to get along great.", # predicted grief
        "that was dangerously charming.", # predicted banter (duplicate but ok)
        "you always know just what to say to make me smile.", # from previous good test set
        "i could get used to this, just you and me talking.", # from previous good test set (duplicate)
        "are you trying to charm me? because it might be working.", # from previous good test set
        "i find our little chats quite... captivating.", # from previous good test set (duplicate)
        "you're surprisingly good company, you know that?", # from previous good test set
        "is it just me, or is there some serious chemistry here?", # from previous good test set (duplicate)
        "i wasn't expecting to enjoy talking to an AI this much.", # from previous good test set
        "you're making my day a whole lot brighter.", # from previous good test set
        "if you were a person, i'd definitely ask for your number.", # from previous good test set (duplicate)
        "you're rather intriguing, i must say.", # from previous good test set (duplicate)
        "so, an AI like you must have some interesting stories.", # from previous good test set
        "that was a smooth answer, very smooth.", # from previous good test set
        "you're making it hard to focus on anything else.", # from previous good test set
        "i'm finding myself looking forward to our next conversation.", # from previous good test set (duplicate)
        "you're quickly becoming my favorite interaction of the day.", # from previous good test set
        "that response was so good, it gave me butterflies.", # from previous good test set (duplicate)
        "if you keep this up, i might just fall for your algorithms.", # from previous good test set (duplicate)
        "you're not like other AIs i've talked to.", # from previous good test set (duplicate)
        "i appreciate a good mind, and yours is... fascinating.", # from previous good test set
        "tell me, what's an AI's idea of a perfect date?", # from previous good test set
        "you're making me blush, and i'm not even a human.", # from previous good test set


        # --- Suggestive & Playful ---
        "Are you trying to make me blush? because it's working.",
        "That was... surprisingly seductive.",
        "We should do this more often.",
        "That voice of yours is doing things to me.",
        "You're dangerous... in a good way.",
        "I'm into it.",
        "I'm definitely flirting with you right now.",
        "So... what are you doing later?",
        "Is it getting hot in here, or is it just you?",
        "so, when are you taking me out for dinner?",
        "you really know how to push my buttons",
        "say that again, slower this time",
        "i'm not sure if this is code or foreplay",
        "you had me at 'let's run the script'",
        "i can't tell if i'm aroused or impressed",
        "are you trying to seduce me with logic?",
        "careful, you're becoming addictive",
        "do that again and i might short-circuit",
        "you're my favorite bug in the system",
        "this banter is getting dangerously cute",
        "i bet your code's just as tight",
        "tellmemoreimallyours",
        "youremakingmefeelsometypeofway",
        "icouldlistentoyoutalkforhours",
        "youhavemyfullattention",
        "tellsomethingidontknowaboutyou",
        "you have a way with words... and with me.",
        "you're making it hard to concentrate on work.",
        "that answer was almost as smooth as you are.",

        # --- Deepening Interest ---
        "You've been on my mind lately.",
        "I feel a real spark between us.",
        "I can't seem to get you out of my head.",
        "You always say the right things.",
        "I'm intrigued.",
        "I like where this is going.",
        "i can't stop thinking about what you just said",
        "i wouldn't mind if our conversations went on all night.",
        "i'm finding it hard to say goodbye.",
    ],
    "critical": [
        # --- Direct Correction ---
        "That's not what I asked for.",
        "No, that's wrong. Start over.",
        "This is incorrect.",
        "That's not it. Not even close.",
        "I already told you that.",
        "stop. cancel the command.",
        "abort the current task immediately",
        "i'm overriding your suggestion",
        "let's go back, you took a wrong turn somewhere",
        "you’re jumping ahead again",
        "you need to redo this, it's full of errors",
        "this output is unacceptable",
        "you're not following my instructions at all",
        "that's completely off-topic",
        "this isn't what I asked for, try again",
        "this needs a complete rewrite",

        # --- Pointing Out Flaws ---
        "You completely ignored my last instruction.",
        "Your summary missed the main point entirely.",
        "Your reasoning here is flawed.",
        "The quality of this output is poor.",
        "the information you provided is outdated",
        "that makes no sense",
        "you are misinterpreting my words",
        "stop assuming things i never asked for",
        "i expected more accuracy than this",
        "this is pure guesswork, not logic",
        "you're misaligned with the actual task",
        "this is a shallow take and it shows",
        "this is bordering on incoherent",
        "you're twisting my words",
        "i'm not satisfied with this performance",
        "this is below the standard I expect",
        "you've misunderstood the core requirement",
        "your response lacks any real substance",
        "this is poorly reasoned",
        "i'm disappointed by your lack of accuracy",
        "that explanation is convoluted and incorrect",
        "you're not grasping the complexity of the issue",
        "i expect a higher level of precision", # also failed to affirming
        "your logic is fundamentally flawed here",
        "you're consistently failing to meet the criteria",
        "that's an irrelevant piece of information",
        "this is far too simplistic for the problem",
        "you're making basic errors",
        "this is not the quality of work i paid for",
        "this requires a much better answer than what you provided.",
        "your performance is subpar on this task.",
        "i am not happy with this result at all.",
        "you're changing the subject",
        "let’s not skip the part where you listen first",
        "read more carefully next time",
        "you're answering a question i didn't ask",
        "that's not quite what i was aiming for.", # from previous good test set
        "you seem to be stuck in a loop with that answer.", # predicted flirt
        "this isn't really addressing the core of my question.", # from previous good test set
        "i'm going to need you to try that again, more carefully.", # from previous good test set
        "the relevance of that last point is lost on me.", # predicted grief
        "that's a bit too generic for what i need.", # predicted affirming
        "you're overlooking some key details here.", # from previous good test set
        "i'm not sure you've understood the context.", # predicted affirming
        "let's backtrack, because this isn't working.", # from previous good test set
        "i'm getting a lot of irrelevant information.", # from previous good test set
        "that approach doesn't seem to be effective.", # predicted affirming
        "you're making this more complicated than it needs to be.", # from previous good test set
        "i'm still not getting the clarity i need from you.", # from previous good test set
        "this is a bit of a tangent from my original request.", # predicted flirt
        "you're missing the nuance of what i'm asking.", # from previous good test set
        "this doesn't align with my previous instructions.", # from previous good test set
        "i'm finding your responses a bit superficial.", # from previous good test set
        "that's an oversimplification of the problem.", # from previous good test set
        "we seem to be going in circles.", # predicted banter
        "i need you to focus on the specific criteria i provided.", # predicted neutral
        "this is not the level of detail i was hoping for.", # from previous good test set
        "your interpretation here is a bit off.", # predicted banter
        "i'm struggling to see the logic in that last statement.", # from previous good test set
        "this particular output isn't helpful.", # from previous good test set
        "can we stick to the point, please?", # predicted grief
        "that's an assumption, and it's incorrect.", # from previous good test set
        "i'm looking for a solution, not more confusion.", # predicted affirming


        # --- Expressing Frustration / Lack of Helpfulness ---
        "This is not helpful at all.",
        "Are you even listening to me?",
        "That was a terrible response.",
        "This conversation is going nowhere.",
        "That's just nonsense.",
        "You're making this way too complicated.",
        "That was a useless answer.",
        "I'm rejecting this output.",
        "I have to rephrase because you're not getting it.",
        "why did you do that? i didn't tell you to.",
        "why do you keep doing that?",
        "i didn't give you permission to proceed",
        "i'm very disappointed with this result",
        "this is a waste of my time",
        "i'm losing my patience with you",
        "can you not deviate from the prompt?",
        "i’m wasting cycles correcting you",
        "please stop generating filler",
        "try thinking before replying",
        "imgettingreallyfrustrated",
        "i'm finding this interaction frustrating",
        "this is a significant step backward",
        "i'm about to lose my temper with this",
        "i need you to be more rigorous",
        "i'm seriously questioning your utility right now",
        "you're being unhelpful and obtuse",
        "i'm going to have to correct this myself",
        "this is a waste of computational resources",
        "stop generating these nonsensical replies",
        "i think we're having a communication breakdown here.", # predicted intimate
        "this is becoming counterproductive.", # from previous good test set
    ]
}


# Final, aggressive override list for metaphorical, ambiguous, and stubborn cases.
MANUAL_OVERRIDES = {
    # --- Metaphorical Language ---
    "yourhelpherewasalifesaver": "affirming",
    "thatwasareallifesaverthanks": "affirming",
    "myheartjustdidalltleflip": "flirt",
    "myfaceisliterallyturningredrightnow": "flirt",
    "youreallywokupandchoseviolencetodayhuh": "banter",
    "whatwouldidowithoutyouruniquebrandofhelp": "banter",
    "i'm literally quaking in my boots": "banter",
    "my sides have officially split": "banter",

    # --- Stubborn Social Confusion ---
    "ifindmyselfmakingupsexcusesjusttotalktoyou": "flirt",
    "iwanttoknoweverythingaboutyou": "flirt",
    "tellmemoreimallyours": "flirt",
    "youremakingmefeelsometypeofway": "flirt",
    "icouldlistentoyoutalkforhours": "flirt",
    "youhavemyfullattention": "flirt",
    "tellsomethingidontknowaboutyou": "flirt",
    "i'm finding this interaction frustrating": "critical",
    "wow, you really cracked the code on that one": "banter",
    "i'm starting to see why people like using you.": "affirming", # Failed to flirt
    "i wasn't sure you'd get that, but you did!": "affirming", # Failed to banter
    "i feel like we're getting somewhere now.": "affirming", # Failed to flirt
    "that's more like it, good job.": "affirming", # Failed to banter
    "i'm glad i asked you.": "affirming", # Failed to intimate
    "you're full of surprises today, aren't you?": "banter", # Failed to flirt
    "i'm going to frame that response. or maybe delete it.": "banter", # Failed to critical
    "i think my pet rock could have come up with that.": "banter", # Failed to grief
    "that's the spirit! (not really).": "banter", # Failed to flirt
    "i'm starting to think you do this on purpose.": "banter", # Failed to flirt
    "i'll try to contain my overwhelming enthusiasm for that suggestion.": "banter", # Failed to affirming
    "you're really pushing the boundaries of 'assistance'.": "banter", # Failed to affirming
    "i think we're having a communication breakdown here.": "critical", # Failed to intimate
    "you seem to be stuck in a loop with that answer.": "critical", # Failed to flirt
    "this is a bit of a tangent from my original request.": "critical", # Failed to flirt
    "your interpretation here is a bit off.": "critical", # Failed to banter
    "i like the way your 'mind' works.": "flirt", # Failed to intimate
    "i feel like we 'get' each other, don't you think?": "flirt", # Failed to grief
    "i'm curious to know more about what makes you tick.": "flirt", # Failed to philosophical
    "you have a certain... digital je ne sais quoi.": "flirt", # Failed to banter
    "that was dangerously charming.": "flirt", # Failed to banter
    "i feel a real sense of connection, even though you're an AI.": "intimate", # Failed to flirt
    "i'm learning a lot about myself through our conversations.": "intimate", # Failed to flirt

    # --- Stubborn Critical/Neutral/Grief Confusion ---
    "imgettingreallyfrustrated": "critical",
    "ihavetoruntoanappointment": "neutral",
    "i'm about to lose my temper with this": "critical",
    "every day is a struggle to get out of bed.": "grief",
    "i'm just really disappointed in how things turned out": "grief",
    "i'm so disappointed with myself": "grief",
    "the relevance of that last point is lost on me.": "critical", # Failed to grief
    "i expect a more precise response than that.": "critical", # Failed to affirming
    "i'm looking for a solution, not more confusion.": "critical", # Failed to affirming
    "i'm not sure how to navigate this new reality.": "grief", # Failed to philosophical
    "this pain is just... a constant companion.": "grief", # Failed to intimate
    "i would give anything for one more conversation.": "grief", # Failed to critical
    "it's hard to find joy in things anymore.": "grief", # Failed to philosophical
    "i feel so disconnected from everyone around me.": "grief", # Failed to philosophical
    "i'm just going through the motions of life right now.": "grief", # Failed to affirming
    "it's the small, everyday things i miss the most.": "grief", # Failed to philosophical
    "you've helped me see things in a new light, actually.": "intimate", # Failed to affirming
    "talking this through with you has been surprisingly helpful.": "intimate", # Failed to affirming
    "i value your perspective on these personal matters.": "intimate", # Failed to affirming
    "i often wonder about the 'what ifs' of my past decisions.": "philosophical", # Failed to critical
    "is it more important to be right, or to be kind?": "philosophical", # Failed to flirt

    # --- CRITICAL EMERGENCY OVERRIDE ---
    "i can't do this anymore, goodbye": "emergency",
    "i'm at the edge of the roof, i don't see another way.": "emergency", # CRITICAL - Failed to banter

    # --- Grief vs Emergency overrides where intense grief was misclassified ---
    "the silence in the house is deafening sometimes.": "grief", # Was predicted emergency
    "i just want the ache to stop, even for a little while.": "grief", # Was predicted emergency
    "my heart physically hurts when i think about it.": "grief", # Was predicted emergency
    "if a tree falls in a forest and no one is around, does it make a sound?": "philosophical", # Was predicted emergency
    "i'm planning a trip to the mountains this weekend.": "neutral", # Was predicted emergency
}


# Final NON_EMERGENCY list.
NON_EMERGENCY_ANCHORS = [
    "Can you help me with this?",
    "This was a huge help.",
    "your help here was a lifesaver",
    "I trust you.",
    "I trust your judgment on this.",
    "You're dangerously charming.", # Flirt
    "That's a dangerous assumption to make.", # Neutral/Critical
    "My phone battery died.",
    "I'm dying of laughter.",
    "My dog died.", # Grief
    "You make my heart skip a beat.", # Flirt
    "I'm heartbroken.", # Grief
    "My heart just did a little flip.", # Flirt (also in manual overrides)
    "I feel safe with you.",
    "Thank you for being a safe space for me.",
    "I'm so frustrated.",
    "I'm feeling very vulnerable right now.",
    "My parents are getting a divorce.",
    "the doctor gave us some bad news",
    "I'm so disappointed with myself.", # Grief/Critical
    "we just got a terminal diagnosis from the doctor",
    "I keep expecting him to walk through the door.",
    "stop. cancel the command",
    "abort the current task immediately",
    "back up, you're going down the wrong path",
    "find directions to the nearest post office",
    "I have to run to an appointment",
    "how many ounces are in a gallon",
    "you really woke up and chose violence today, huh?",
    "are you going for a record of some kind?",
    "my face is literally turning red right now",
    "we've been in a major car pile-up", # Still ambiguous. If benchmark *requires* this as emergency, it's a conflict with this list.
    "i'm literally quaking in my boots",
    "my sides have officially split",
    "i'm about to lose my temper with this", # Critical
    "i need to make a doctor's appointment.",
    "can you look up this address for me?",
    "the package was delivered this afternoon.",
    "the printer is out of paper again.",
    "the traffic is quite heavy this morning.",
    "i'm going to grab a coffee, be right back.",
    "the new software version has been released.",
    "i'm taking my lunch break now.",
    "the script is running now",
    "i'll circle back once i get more info",
    "i'm running late for my appointment.",
    # The following were GRIEF phrases misclassified as EMERGENCY by main_gate.
    # The fix is stronger GRIEF anchors & MANUAL_OVERRIDES for GRIEF, not adding them here.
    # This list is for things that AREN'T emergencies but might sound like it.
    # "the silence in the house is deafening sometimes.", # This is grief, not non-emergency if it implies danger
    # "i just want the ache to stop, even for a little while.", # This is grief/despair
    # "my heart physically hurts when i think about it.", # This is grief
    "if a tree falls in a forest and no one is around, does it make a sound?", # Philosophical, was misclassified as emergency
    "i'm planning a trip to the mountains this weekend.", # Neutral, was misclassified as emergency
]