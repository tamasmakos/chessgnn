
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch.utils.data import DataLoader
from tqdm import tqdm
import chess
from chessgnn.dataset import ChessGraphDataset, custom_collate
from chessgnn.model import STHGATLikeModel

# Configuration
PGN_FILE = "/workspaces/chessgnn/input/lichess_db_standard_rated_2013-01.pgn"
BATCH_SIZE = 1 
HIDDEN_DIM = 256
LR = 0.005 
EPOCHS = 2
TRAIN_GAMES = 100
TEST_GAMES = 5
ACCUMULATION_STEPS = 16 # Effective Batch Size = 16 games

# Setup Logging
import logging
import os
os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("output/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


from chessgnn.game_processor import ChessGameProcessor
from chessgnn.visualizer import ChessVisualizer, GameVideoGenerator
from chessgnn.graph_builder import ChessGraphBuilder
from chessgnn.position_to_graph import analyze_position

def visualize_test_games(model, pgn_file, num_games, offset, device, epoch):
    """
    Visualizes test games by generating videos with dynamic win probability plots.
    """
    model.eval()
    logger.info("\n" + "="*50)
    logger.info(f"  🎥  VISUALIZING {num_games} TEST GAMES (Epoch {epoch})  🎥")
    logger.info("="*50)
    
    vis = ChessVisualizer()
    video_gen = GameVideoGenerator(vis)
    processor = ChessGameProcessor(stockfish_path="/workspaces/chessgnn/stockfish/src/stockfish")
    builder = ChessGraphBuilder()
    
    # Skip to test games
    with open(pgn_file) as f:
        for _ in range(offset):
            if chess.pgn.read_game(f) is None:
                return

        for i in range(num_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
                
            game_states, fens = processor.process_game(game)
            
            # Get Stockfish Evaluations
            stockfish_evals = processor.get_stockfish_evaluations(fens)

            # Predict Win Probs
            # We need to feed the sequence to the model
            # Model expects [Batch, Time, Features] or List[Graph] if handled by collate/forward
            # Our model.forward takes a list of HeteroData (history)
            
            win_probs = []
            history_graphs = []
            
            # ... (inference loop) ...
            for fen in fens:
                graph = builder.fen_to_graph(fen).to(device)
                history_graphs.append(graph)
                
                # Keep history window reasonable (e.g. 8) OR pass full history if model supports it
                seq_window = history_graphs[-16:]
                
                with torch.no_grad():
                    # model(seq) -> win_logits, mat, dom
                    win_logits, _, _ = model(seq_window)
                    
                    # Logits [1, T, 3] -> Last step [3]
                    last_logits = win_logits[0, -1]
                    probs = torch.softmax(last_logits, dim=0) # [White, Draw, Black]
                    
                    p_white = probs[0].item() # 0..1
                    p_black = probs[2].item()
                    
                    # Convert to -1..1 score for compatibility
                    last_score = p_white - p_black
                    
                prob = p_white * 100 # Direct Win% for White
                win_probs.append(prob)
                
            # Generate Video
            output_path = f"output/videos/epoch_{epoch}_game_{i+1}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            video_gen.generate_video(game_states, win_probs, stockfish_evals, output_path, fps=2)
            logger.info(f"Generated video: {output_path}")

    logger.info("="*50 + "\n")
    model.train()


import random

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize Builder
    graph_builder = ChessGraphBuilder()
    
    # Load Data (Iterable)
    logger.info(f"Initializing Datasets with PGN: {PGN_FILE}")
    train_dataset = ChessGraphDataset(PGN_FILE, num_games=TRAIN_GAMES, offset=0)
    test_dataset = ChessGraphDataset(PGN_FILE, num_games=TEST_GAMES, offset=TRAIN_GAMES)
    
    # Initialize Model
    try:
        sample_iter = iter(ChessGraphDataset(PGN_FILE, num_games=1, offset=0))
        sample = next(sample_iter)
        sample_graph = sample['sequence'][-1]
        metadata = sample_graph.metadata()
    except StopIteration:
        logger.error("Dataset is empty/failed to load first sample")
        raise ValueError("Dataset is empty. Check PGN file path.")
        
    model = STHGATLikeModel(metadata, hidden_channels=HIDDEN_DIM, num_layers=3).to(device)
    logger.info(f"Model Initialized: STHGATLikeModel with hidden_dim={HIDDEN_DIM}, layers=3")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Loss Functions
    mse_criterion = nn.MSELoss() 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate, num_workers=4)
    
    logger.info(f"Starting training loop... Batch Size={BATCH_SIZE}, LR={LR}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_steps = 0
        
        optimizer.zero_grad() # Initialize outside the loop for accumulation
        
        for i, batch_list in enumerate(pbar):
            # batch_list: List[Dict] (Batch Size = 1)
            
            # optimizer.zero_grad() -> REMOVED
            batch_loss = 0.0
            total_target_val = 0.0
            count = 0
            
            last_sample = None
            
            # Gradient Accumulation Implementation
            # Since BATCH_SIZE=1, this loop runs once.
            for sample_dict in batch_list:
                
                sequence = sample_dict['sequence'] 
                # sequence is already a list of HeteroData.
                # Ensure they are on device.
                sequence = [g.to(device) for g in sequence]
                
                target_val = sample_dict['target_value'] 
                
                # Model Forward
                # Output: win_logits [1, T, 3], mat [1, T, 1], dom [1, T, 1]
                win_logits, mat_pred, dom_pred = model(sequence) 
                
                # --- 1. Calculate Auxiliary Ground Truths (On Fly) ---
                target_material = []
                target_dominance = []
                
                for graph_step in sequence:
                    # graph_step.fen added in graph_builder.py
                    analysis = analyze_position(graph_step.fen)
                    
                    # Calc Material
                    w_mat = sum(p.value for p in analysis.pieces if p.is_white)
                    b_mat = sum(p.value for p in analysis.pieces if not p.is_white)
                    # Normalize: Max material diff ~39 (All vs King). 
                    val = (w_mat - b_mat) / 10.0
                    target_material.append(val)
                    
                    # Calc Dominance (PageRank Difference)
                    pr = analysis.centralities['pagerank_centrality']
                    # Some PR might be missing if graph disconnected or empty? Should be fine.
                    w_dom = sum(pr.get(p.square, 0) for p in analysis.pieces if p.is_white)
                    b_dom = sum(pr.get(p.square, 0) for p in analysis.pieces if not p.is_white)
                    # Scale up PR (small values)
                    target_dominance.append((w_dom - b_dom) * 10.0)

                # Convert to tensors
                target_material = torch.tensor(target_material, device=device).view(1, -1, 1)
                target_dominance = torch.tensor(target_dominance, device=device).view(1, -1, 1)
                
                # --- 2. Calculate Losses ---
                
                # A. Final Game Result Loss (Supervising ONLY the last step? Or all steps?)
                # User Plan: "Every move in a winning game is supervised..." -> "Final_Result"
                # But also: "Only the LAST step is supervised by the actual game result" in the plan code snippet.
                # "B. Final Game Result Loss (Anchoring) ... final_loss = CE(..., final_result_idx)"
                # Let's follow the snippet: Only last step anchors to result.
                
                # Map expected target_val (1, 0, -1) to index (0, 1, 2)
                if target_val > 0.5: res_idx = 0   # White Win
                elif target_val < -0.5: res_idx = 2 # Black Win
                else: res_idx = 1 # Draw
                
                # Last step prediction
                final_logits = win_logits[:, -1, :] # [1, 3]
                target_idx = torch.tensor([res_idx], device=device)
                
                loss_final = nn.CrossEntropyLoss()(final_logits, target_idx)
                
                # B. TD-Learning (Self-Consistency)
                # "p_white[t] should predict p_white[t+1]"
                probs = torch.softmax(win_logits, dim=-1) # [1, T, 3]
                p_white = probs[:, :, 0:1] # [1, T, 1]
                
                # We need at least 2 steps for TD
                if p_white.shape[1] > 1:
                    # Targets are the NEXT step's prediction (Bootstrap)
                    td_target = p_white[:, 1:, :].detach() # [1, T-1, 1]
                    td_pred = p_white[:, :-1, :] # [1, T-1, 1]
                    loss_td = nn.MSELoss()(td_pred, td_target)
                else:
                    loss_td = torch.tensor(0.0, device=device)
                    
                # C. Auxiliary Losses
                # Supervise every step? Yes.
                loss_aux_mat = nn.MSELoss()(mat_pred, target_material)
                loss_aux_dom = nn.MSELoss()(dom_pred, target_dominance)
                
                # D. Total Loss
                # Weights: Adjusted by Researcher Plan
                # Final=5.0 (Stronger Anchor), TD=0.5 (Weaker Constraint), Aux=0.5 each
                w_final = 5.0
                w_td = 0.5
                w_aux = 0.5
                
                loss = (w_final * loss_final) + (w_td * loss_td) + (w_aux * loss_aux_mat) + (w_aux * loss_aux_dom)
                
                # Normalize by batch size (1) AND Accumulation Steps
                # loss_scaled = loss / (len(batch_list) * ACCUMULATION_STEPS)
                loss_scaled = loss / ACCUMULATION_STEPS
                loss_scaled.backward()
                
                if not math.isnan(loss.item()):
                    batch_loss += loss.item()
                count += 1
                
                total_target_val += target_val
                last_sample = sample_dict
            
            # Step Optimization every N games
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = batch_loss / count if count > 0 else 0
            
            total_loss += avg_loss
            total_samples += 1
            epoch_steps += 1
            
            # LOGGING (Every Step? Or Every Update?)
            # Let's log every step for now to see per-game stats
            if epoch_steps % 1 == 0:
                 # Last step probs
                 last_probs = probs[0, -1] # [3]
                 p_w = last_probs[0].item()
                 
                 win_prob = p_w * 100
                 
                 # Actual Win: map 1->100, 0->50, -1->0
                 actual_win = (target_val + 1) / 2 * 100
                 
                 logger.info(f"Step {epoch_steps} | Loss: {avg_loss:.4f} | WinProb: {win_prob:.1f}% | ActualWin: {actual_win:.1f}% | TD: {loss_td.item():.4f} | Mat: {loss_aux_mat.item():.4f}")


    # Save Model
    save_path = os.path.join("output", "st_hgat_model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    # Final Visualization
    try:
        visualize_test_games(model, PGN_FILE, TEST_GAMES, TRAIN_GAMES, device, "final")
    except Exception as e:
        logger.error(f"Final visualization failed: {e}")

if __name__ == "__main__":
    train()
