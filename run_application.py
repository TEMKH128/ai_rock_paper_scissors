def determine_winner(player_move, comp_move):
    """
    Determines the winner of the rock-paper-scissors game between
    player and computer.
    Parameters:
      * user_move: move executed by user (R/P/S).
      * player_move: move executed by computer (R/P/S).
    Return: outcome of move - tie/computer/player.
    """
    player_move = player_move.lower()
    comp_move = comp_move.lower()

    if (player_move == comp_move):
        return "tie"

    # Computer wins.
    elif ((player_move == "rock" and comp_move == "scissors") or
        (player_move == "paper" and comp_move == "rock") or
        (player_move == "scissors" and comp_move == "paper")):

        return "player"
    
    else:  ## Computer wins.
        return "computer"

    
