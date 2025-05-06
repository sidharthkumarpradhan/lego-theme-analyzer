import pandas as pd
import numpy as np
import streamlit as st
import traceback
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from logger_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

class ThemePredictor:
    """
    Machine learning model to predict LEGO set themes based on parts
    """
    def __init__(self):
        """
        Initialize theme predictor
        """
        self.model = None
        self.theme_names = {}
        self.feature_names = None
        
    def train(self, X, y, theme_names=None, memory_efficient=False):
        """
        Train the theme prediction model
        
        Args:
            X (DataFrame): Feature matrix where each row is a set and columns are parts
            y (Series): Target vector with theme_ids
            theme_names (dict): Mapping from theme_id to theme_name
            memory_efficient (bool): Whether to use memory-efficient training approach
            
        Returns:
            bool: Whether training was successful
        """
        logger.info("Starting theme predictor training")
        logger.info(f"Memory efficient mode: {memory_efficient}")
        
        if X is None or y is None:
            error_msg = "No data available for training"
            logger.error(error_msg)
            st.error(error_msg)
            return False
            
        # Save feature names (part numbers) for later reference
        self.feature_names = X.columns.tolist()
            
        try:
            # Log initial data shape
            initial_shape = f"Initial data: {X.shape[0]} sets with {X.shape[1]} parts"
            logger.info(initial_shape)
            st.info(initial_shape)
            
            # Log memory usage of feature matrix
            memory_usage_mb = X.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"Feature matrix memory usage: {memory_usage_mb:.2f} MB")
            
            # If memory efficient mode, reduce dataset size further for training
            if memory_efficient:
                logger.info("Applying memory optimization techniques to reduce dataset size")
                
                # Reduce features by selecting only more common parts
                # This drastically reduces memory usage with minimal impact on prediction quality
                
                # Sum up each column (part) to get total usage counts
                logger.info("Calculating part usage frequencies")
                part_usage = X.sum().sort_values(ascending=False)
                
                # Get most used parts (top 1000 or fewer if less are available)
                max_parts = min(1000, len(part_usage))
                most_used_parts = part_usage.head(max_parts).index.tolist()
                logger.info(f"Selected top {len(most_used_parts)} most frequently used parts")
                
                # Filter X to only include most used parts
                X_original_shape = X.shape
                X = X[most_used_parts]
                logger.info(f"Reduced features from {X_original_shape[1]} to {X.shape[1]}")
                
                # Filter out rare themes (those with only 1-2 sets)
                logger.info("Filtering out rare themes")
                theme_counts = y.value_counts()
                common_themes = theme_counts[theme_counts >= 3].index
                logger.info(f"Selected {len(common_themes)} themes with 3+ sets (out of {len(theme_counts)} total themes)")
                
                # Create mask for common themes
                mask = y.isin(common_themes)
                
                # Filter both X and y to only include common themes
                original_rows = len(X)
                X = X.loc[mask]
                y = y.loc[mask]
                logger.info(f"Reduced samples from {original_rows} to {len(X)} after removing rare themes")
                
                # Log memory usage after reduction
                reduced_memory_mb = X.memory_usage(deep=True).sum() / 1024 / 1024
                logger.info(f"Memory usage after reduction: {reduced_memory_mb:.2f} MB (saved {memory_usage_mb - reduced_memory_mb:.2f} MB)")
                
                memory_optimized_msg = f"Memory-optimized training data: {X.shape[0]} sets with {X.shape[1]} parts"
                logger.info(memory_optimized_msg)
                st.info(memory_optimized_msg)
            
            # Save feature names
            self.feature_names = X.columns.tolist()
            logger.info(f"Saved {len(self.feature_names)} feature names for future prediction")
            
            # Split data
            logger.info("Splitting data into training and test sets (80/20)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            # Make sure we have enough samples and features for PCA
            n_samples = X.shape[0]
            n_features = X.shape[1]
            logger.info(f"Total samples: {n_samples}, Total features: {n_features}")
            
            # Use a more conservative n_components value to avoid errors
            # It must be between 0 and min(n_samples, n_features)
            safe_n_components = min(6, n_samples - 1, n_features - 1) if n_samples > 1 and n_features > 1 else 2
            logger.info(f"Using PCA with {safe_n_components} components for dimensionality reduction")
            
            # Adjust model complexity based on memory efficiency setting
            if memory_efficient:
                logger.info("Creating memory-efficient model pipeline")
                # Use more memory-efficient model configuration
                self.model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=safe_n_components)),
                    ('classifier', RandomForestClassifier(
                        n_estimators=50,  # Reduced number of trees
                        max_depth=10,     # Limit tree depth
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=2,         # Limit parallel jobs
                        class_weight='balanced'
                    ))
                ])
                logger.info("Memory-efficient model: RandomForest with 50 trees, max_depth=10, n_jobs=2")
            else:
                logger.info("Creating standard model pipeline")
                # Original configuration
                self.model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=safe_n_components)),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'
                    ))
                ])
                logger.info("Standard model: RandomForest with 100 trees, unlimited depth, n_jobs=-1")
            
            # Check if we have enough data for meaningful train/test split
            if len(X_train) < 2 or len(X_test) < 2:
                warning_msg = "Not enough data for train/test split. Using all data for training."
                logger.warning(warning_msg)
                st.warning(warning_msg)
                
                # Use all data for training
                logger.info("Fitting model on entire dataset")
                self.model.fit(X, y)
                
                success_msg = "Theme prediction model trained with limited data"
                logger.info(success_msg)
                st.success(success_msg)
            else:
                # Train model with train/test split
                logger.info("Fitting model on training data")
                
                # Measure training time
                import time
                start_time = time.time()
                
                self.model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                logger.info(f"Model training completed in {training_time:.2f} seconds")
                
                # Evaluate
                try:
                    logger.info("Evaluating model on test data")
                    y_pred = self.model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Log more detailed metrics
                    logger.info(f"Model accuracy: {accuracy:.4f}")
                    
                    # Get classification report but only log to console, not UI
                    report = classification_report(y_test, y_pred)
                    logger.info(f"Classification report:\n{report}")
                    
                    # Get confusion matrix shape for sanity check
                    unique_themes = len(np.unique(y))
                    logger.info(f"Model trained to predict {unique_themes} unique themes")
                    
                    success_msg = f"Theme prediction model trained successfully with {accuracy:.2f} accuracy"
                    logger.info(success_msg)
                    st.success(success_msg)
                except Exception as e:
                    error_details = traceback.format_exc()
                    logger.error(f"Could not evaluate model: {str(e)}\n{error_details}")
                    st.warning(f"Could not evaluate model: {e}")
                    st.success("Theme prediction model trained but not evaluated")
            
            # Save theme names mapping
            self.theme_names = theme_names or {}
            logger.info(f"Saved mapping for {len(self.theme_names)} theme names")
            
            # Free up some memory
            if memory_efficient:
                logger.info("Running garbage collection to free memory")
                import gc
                gc.collect()
                
            logger.info("Theme predictor training completed successfully")
            return True
            
        except Exception as e:
            error_details = traceback.format_exc()
            error_msg = f"Error training theme prediction model: {str(e)}"
            logger.error(f"{error_msg}\n{error_details}")
            st.error(error_msg)
            return False
    
    def predict(self, parts_dict):
        """
        Predict theme for a set of parts
        
        Args:
            parts_dict (dict): Dictionary mapping part_num to quantity
            
        Returns:
            dict: Prediction results with theme_id, theme_name, and confidence
        """
        logger.info("Starting theme prediction")
        
        if self.model is None:
            error_msg = "Model not trained. Please train the model first."
            logger.error(error_msg)
            st.error(error_msg)
            return None
            
        try:
            # Log parts dict summary
            n_parts = len(parts_dict)
            total_quantity = sum(parts_dict.values())
            logger.info(f"Predicting theme for {n_parts} unique parts (total quantity: {total_quantity})")
            
            # Create feature vector
            logger.info(f"Creating feature vector with {len(self.feature_names)} columns")
            features = pd.DataFrame(0, index=[0], columns=self.feature_names)
            
            # Track parts that are used in the model
            matched_parts = 0
            matched_quantity = 0
            
            # Fill in available parts
            for part_num, quantity in parts_dict.items():
                if part_num in self.feature_names:
                    features.loc[0, part_num] = quantity
                    matched_parts += 1
                    matched_quantity += quantity
            
            # Log parts matching statistics
            match_percent = 100 * matched_parts / n_parts if n_parts > 0 else 0
            logger.info(f"Parts matching model features: {matched_parts}/{n_parts} ({match_percent:.1f}%)")
            logger.info(f"Quantity matching model features: {matched_quantity}/{total_quantity} " 
                       f"({100 * matched_quantity / total_quantity if total_quantity > 0 else 0:.1f}%)")
            
            # If very few parts match, log a warning
            if matched_parts < 5 and n_parts > 10:
                logger.warning(f"Very few parts match model features ({matched_parts}/{n_parts}). "
                              f"Prediction may be unreliable.")
            
            # Get prediction and probabilities
            logger.info("Running prediction through model pipeline")
            start_time = time.time()
            theme_id = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            prediction_time = time.time() - start_time
            logger.info(f"Prediction completed in {prediction_time:.4f} seconds")
            
            # Get confidence of prediction
            confidence = probabilities.max()
            logger.info(f"Predicted theme ID: {theme_id} with confidence: {confidence:.4f}")
            
            # Get theme name
            theme_name = self.theme_names.get(theme_id, f"Theme {theme_id}")
            logger.info(f"Predicted theme name: {theme_name}")
            
            # Get top 3 theme predictions
            logger.info("Identifying top 3 theme predictions")
            top_indices = probabilities.argsort()[-3:][::-1]
            top_themes = []
            
            for i, idx in enumerate(top_indices):
                theme_id_i = self.model.classes_[idx]
                theme_name_i = self.theme_names.get(theme_id_i, f"Theme {theme_id_i}")
                confidence_i = probabilities[idx]
                
                logger.info(f"Top {i+1}: Theme {theme_id_i} ({theme_name_i}) - Confidence: {confidence_i:.4f}")
                
                top_themes.append({
                    'theme_id': theme_id_i,
                    'theme_name': theme_name_i,
                    'confidence': confidence_i
                })
            
            # Create result dictionary
            result = {
                'theme_id': theme_id,
                'theme_name': theme_name,
                'confidence': confidence,
                'top_themes': top_themes
            }
            
            logger.info("Theme prediction completed successfully")
            return result
            
        except Exception as e:
            error_details = traceback.format_exc()
            error_msg = f"Error making theme prediction: {str(e)}"
            logger.error(f"{error_msg}\n{error_details}")
            st.error(error_msg)
            return None
    
    def get_feature_importance(self, n_top=20):
        """
        Get the most important parts for theme prediction
        
        Args:
            n_top (int): Number of top features to return
            
        Returns:
            DataFrame: Top parts with importance scores
        """
        logger.info(f"Extracting feature importance (top {n_top} components)")
        
        if self.model is None:
            logger.error("Cannot get feature importance - model not trained")
            return pd.DataFrame()
            
        if not hasattr(self.model['classifier'], 'feature_importances_'):
            logger.error("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        try:    
            # Get feature importances
            logger.info("Extracting feature importances from classifier")
            importances = self.model['classifier'].feature_importances_
            logger.info(f"Found {len(importances)} importance values")
            
            # Get the PCA components
            if hasattr(self.model['pca'], 'components_'):
                # PCA components matrix has shape (n_components, n_features)
                pca_components = self.model['pca'].components_
                
                # Create dataframe for PC importance with additional information
                pc_df = pd.DataFrame({
                    'PC': [f'PC{i+1}' for i in range(len(importances))],
                    'Importance': importances,
                    'KeyParts': [''] * len(importances)
                }).sort_values('Importance', ascending=False).head(n_top)
                
                # For each principal component, find the most influential parts
                for i, pc_idx in enumerate(pc_df.index):
                    pc_num = pc_df.loc[pc_idx, 'PC']
                    
                    # Extract the component number from 'PC1', 'PC2', etc.
                    try:
                        pc_num_int = int(pc_num.replace('PC', '')) - 1  # Convert to 0-based index
                        logger.info(f"Processing PC {pc_num} (index {pc_num_int})")
                        
                        if pc_num_int < len(pca_components):
                            # Get the component weight vector
                            weights = pca_components[pc_num_int]
                            
                            # Get top parts for this component (highest absolute weights)
                            top_part_indices = np.argsort(np.abs(weights))[-3:]  # Top 3 parts
                            top_part_indices = top_part_indices[::-1]  # Reverse to get highest first
                            
                            # Get part numbers for these indices
                            if self.feature_names and len(self.feature_names) > 0:
                                top_parts = []
                                for idx in top_part_indices:
                                    if idx < len(self.feature_names):
                                        top_parts.append(self.feature_names[idx])
                                    else:
                                        top_parts.append("Unknown")
                                
                                # Add to dataframe with importance scores
                                top_parts_with_scores = [
                                    f"{part} ({abs(weights[idx]):.3f})" 
                                    for part, idx in zip(top_parts, top_part_indices)
                                ]
                                
                                # Add to dataframe
                                pc_df.loc[pc_idx, 'KeyParts'] = ", ".join(top_parts_with_scores)
                                logger.info(f"Added key parts for {pc_num}: {pc_df.loc[pc_idx, 'KeyParts']}")
                            else:
                                logger.warning(f"Feature names unavailable when processing {pc_num}")
                                pc_df.loc[pc_idx, 'KeyParts'] = "Unknown parts"
                        else:
                            logger.warning(f"PC index {pc_num_int} out of bounds for components matrix with shape {pca_components.shape}")
                            pc_df.loc[pc_idx, 'KeyParts'] = "Out of bounds"
                    except Exception as e:
                        logger.error(f"Error processing PC {pc_num}: {str(e)}")
                        pc_df.loc[pc_idx, 'KeyParts'] = "Error"
                
                # Log top 5 most important components
                top5 = pc_df.head(5)
                logger.info(f"Top 5 most important components with key parts:\n{top5.to_string()}")
                
                # Calculate total importance captured by top components
                total_importance = importances.sum()
                top_importance = pc_df['Importance'].sum()
                logger.info(f"Top {n_top} components capture {100 * top_importance / total_importance:.2f}% of total importance")
                
                # For display purposes, rename the PC column to include key parts
                pc_df['PC_Display'] = pc_df.apply(
                    lambda row: f"{row['PC']} ({row['KeyParts']})", axis=1
                )
                
                # Return the enhanced dataframe
                logger.info(f"Returning feature importance dataframe with {len(pc_df)} rows and part information")
                return pc_df
            else:
                # If PCA components aren't available, fallback to basic PC representation
                logger.warning("PCA components not available - falling back to basic PC representation")
                pc_importance = pd.DataFrame({
                    'PC': [f'PC{i+1}' for i in range(len(importances))],
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(n_top)
                
                return pc_importance
            
            # Log top 5 most important components
            top5 = pc_importance.head(5)
            logger.info(f"Top 5 most important components:\n{top5.to_string()}")
            
            # Calculate total importance captured by top components
            total_importance = importances.sum()
            top_importance = pc_importance['Importance'].sum()
            logger.info(f"Top {n_top} components capture {100 * top_importance / total_importance:.2f}% of total importance")
            
            logger.info(f"Returning feature importance dataframe with {len(pc_importance)} rows")
            return pc_importance
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error getting feature importance: {str(e)}\n{error_details}")
            return pd.DataFrame()
