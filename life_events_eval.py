import json
import argparse

def analyze_life_events(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    life_events_count = 0
    sentences_with_events = 0
    event_types = {}
    temporal_calculations = 0
    
    print("Life Events Analysis")
    print("=" * 50)
    
    for i, result in enumerate(results):
        sentence = result['sentence']
        extracted = result['extracted']
        
        family_members = extracted.get('family_members', [])
        sentence_events = 0
        
        print(f"\nSentence {i+1}: {sentence[:60]}...")
        
        for member in family_members:
            if member.get('name') == 'narrator':
                continue
                
            life_events = member.get('life_events', [])
            if life_events:
                sentence_events += len(life_events)
                
                print(f"\n  {member.get('name')} has {len(life_events)} life events:")
                
                for event in life_events:
                    event_type = event.get('event_type')
                    if event_type:
                        event_types[event_type] = event_types.get(event_type, 0) + 1
                    
                    # Check for temporal calculation
                    has_calculation = False
                    start_year = event.get('start_year')
                    end_year = event.get('end_year')
                    duration = event.get('duration')
                    time_period = event.get('time_period')
                    
                    if (start_year and end_year and duration) or (start_year and end_year and time_period):
                        has_calculation = True
                        temporal_calculations += 1
                    
                    # Display event details
                    print(f"    • {event_type}")
                    
                    for k in event:
                        v = event[k]
                        if k != 'event_type' and v:
                            print(f"      - {k}: {v}")
                    
                    if has_calculation:
                        print(f"      (Includes temporal calculation)")
            
            life_events_count += len(life_events)
        
        if sentence_events > 0:
            sentences_with_events += 1
    
    print("\nSummary:")
    print(f"Total life events extracted: {life_events_count}")
    print(f"Sentences with life events: {sentences_with_events}/{len(results)}")
    print(f"Temporal calculations performed: {temporal_calculations}")
    
    # Event type table
    print("\nEvent Types Distribution:")
    for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        percentage = f"{count/life_events_count*100:.1f}%"
        print(f"{event_type}: {count} ({percentage})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze life events in extraction results")
    parser.add_argument("file_path", help="Path to extraction_results.json file")
    args = parser.parse_args()
    
    analyze_life_events(args.file_path)
