# from flask_wtf import FlaskForm
# from wtforms import StringField, PasswordField, SubmitField, BooleanField
# from wtforms.validators import DataRequired, Length, Email, EqualTo


# class MusicForm(FlaskForm):
#     song_name = StringField('Song_name',
#                            validators=[DataRequired(), Length(min=2, max=20)])
#     song_year = StringField('Year',
#                            validators=[DataRequired(), Length(min=1, max=20)])
#     submit = SubmitField('Sign Up')


from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import DataRequired, Length


class MusicForm(FlaskForm):
    song_name = StringField('Song Name',
                           validators=[DataRequired(), Length(min=2, max=20)])
    song_year = StringField('Year Released',
                           validators=[DataRequired(), Length(min=1, max=20)])
    submit = SubmitField('Find Similar Songs')