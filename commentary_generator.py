import random

class CommentaryGenerator:
    def __init__(self):
        # Define templates for different event types
        self.templates = {
            'pass': [
                "A smooth pass by player {player_id}.",
                "Player {player_id} threads the ball to a teammate.",
                "An accurate pass completed by player {player_id}.",
                "Player {player_id} shows great vision with that pass."
            ],
            'lost possession': [
                "Player {player_id} loses the ball.",
                "A turnover by player {player_id}.",
                "Player {player_id} gives up possession.",
                "Unfortunate loss of control by player {player_id}."
            ],
            'contested': [
                "There's a scramble for the ball!",
                "Both teams fighting hard for possession.",
                "A contested ball situation arises!",
                "Tension builds as players clash for the ball."
            ],
            'goal': [
                "Goal! Brilliant finish by player {player_id}!",
                "The ball hits the back of the net — goal by player {player_id}!",
                "An outstanding goal scored by player {player_id}!",
                "Player {player_id} finds the net with a powerful strike!"
            ],
            'save': [
                "What a save by the goalkeeper!",
                "An excellent stop keeps the ball out.",
                "The goalkeeper denies the goal with a fantastic save.",
                "Brilliant reflexes from the keeper!"
            ]
        }

    def generate_commentary(self, event_type, player_id=None):
        # Choose a random template for the event
        if event_type not in self.templates:
            return None

        template = random.choice(self.templates[event_type])

        # Fill in the player ID if needed
        if "{player_id}" in template and player_id is not None:
            commentary = template.format(player_id=player_id)
        else:
            commentary = template

        return commentary


# Test example (you can delete this in final project):
if __name__ == "__main__":
    cg = CommentaryGenerator()
    print(cg.generate_commentary('pass', player_id=10))


